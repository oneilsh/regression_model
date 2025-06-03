import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from google.cloud import bigquery
import yaml
import os

# Mock environment variables for local testing if not in AoU Workbench
# os.environ["WORKSPACE_CDR"] = "all-of-us-research-workbench-####.r2023q3_unzipped_data"
# os.environ["GOOGLE_CLOUD_PROJECT"] = "your-gcp-project-id"

def load_configuration(config_filepath: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found: {config_filepath}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing configuration YAML: {e}")

def get_aou_cdr_path() -> str:
    """Returns the base path for the All of Us Controlled Tier Dataset."""
    if "WORKSPACE_CDR" not in os.environ:
        raise EnvironmentError("WORKSPACE_CDR environment variable not set. "
                               "This is required in the All of Us Workbench.")
    return os.environ["WORKSPACE_CDR"]

def build_cohort_criteria_sql(concepts: List[Dict[str, Any]], cdr_path: str, operator: str = "IN") -> Optional[str]:
    """
    Builds SQL fragments for cohort inclusion/exclusion based on concepts.
    Assumes `concept_id` and `domain` are provided for each concept.
    This uses a simplified EXISTS check against `cb_search_all_events`.
    """
    if not concepts:
        return None

    domain_conditions = []
    for concept_entry in concepts:
        concept_id = concept_entry.get('concept_id')
        # We don't strictly need the 'domain' here if `cb_search_all_events` is comprehensive
        # for all event types. However, for robustness, one could check if concept_id
        # is a valid integer.
        if concept_id is None:
            raise ValueError(f"Concept in cohort_definition must have 'concept_id': {concept_entry}")

        # The 'operator' is either "IN" or "NOT IN" for the concept_id list
        domain_conditions.append(f"""
            EXISTS (
                SELECT 1
                FROM `{cdr_path}.cb_search_all_events` csae
                WHERE csae.person_id = person.person_id
                AND csae.concept_id {operator} ({concept_id})
            )
        """)
    return " OR ".join(domain_conditions)

def build_person_base_query(config: Dict[str, Any]) -> str:
    """
    Builds the base SQL query for the person table, incorporating cohort definition
    from the configuration.
    """
    cdr_path = get_aou_cdr_path()
    
    select_clauses = [
        "person.person_id",
        "FLOOR(DATE_DIFF(DATE(CURRENT_DATE),DATE(person.birth_datetime), DAY)/365.25) AS current_age",
        "person.gender_concept_id",
        "p_gender_concept.concept_name as gender",
        "person.birth_datetime as date_of_birth",
        "person.race_concept_id",
        "p_race_concept.concept_name as race",
        "person.ethnicity_concept_id",
        "p_ethnicity_concept.concept_name as ethnicity",
        "person.sex_at_birth_concept_id",
        "p_sex_at_birth_concept.concept_name as sex_at_birth",
        "cb_search_person.age_at_consent as age_at_consent",
        "cb_search_person.has_ehr_data as has_ehr_data",
        "CASE WHEN p_observation.observation_concept_id = 1586099 AND p_observation.value_source_value = 'ConsentPermission_Yes' THEN 'Yes' ELSE 'No' END AS ehr_consent",
        "person.year_of_birth"
    ]

    from_join_clauses = [
        f"FROM `{cdr_path}.person` person",
        f"LEFT JOIN `{cdr_path}.concept` p_gender_concept ON person.gender_concept_id = p_gender_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_race_concept ON person.race_concept_id = p_race_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_ethnicity_concept ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_sex_at_birth_concept ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.cb_search_person` cb_search_person ON person.person_id = cb_search_person.person_id",
        f"LEFT JOIN `{cdr_path}.observation` p_observation ON person.person_id = p_observation.person_id AND p_observation.observation_concept_id = 1586099"
    ]

    where_conditions = []
    cohort_def = config.get('cohort_definition', {})

    cohort_table_id = cohort_def.get('cohort_table_id')
    if cohort_table_id:
        where_conditions.append(f"person.person_id IN (SELECT person_id FROM `{cdr_path}.{cohort_table_id}`)")
    else:
        include_concepts = cohort_def.get('include_concepts', [])
        exclude_concepts = cohort_def.get('exclude_concepts', [])

        include_sql = build_cohort_criteria_sql(include_concepts, cdr_path, operator="IN")
        # For exclusion, we want to ensure *none* of the exclude concepts are present.
        # This means an AND NOT EXISTS for each exclusion criteria.
        # A simpler way is to build the exclude SQL using 'IN' and then wrap it with NOT EXISTS for the person.
        exclude_person_sql = build_cohort_criteria_sql(exclude_concepts, cdr_path, operator="IN")


        if include_sql:
            where_conditions.append(f"({include_sql})")
        if exclude_person_sql:
            # We need to explicitly check if the person has *none* of the excluded concepts
            # using NOT EXISTS on a subquery for person_id
            where_conditions.append(f"""
                person.person_id NOT IN (
                    SELECT DISTINCT csae_excl.person_id
                    FROM `{cdr_path}.cb_search_all_events` csae_excl
                    WHERE csae_excl.concept_id IN ({','.join(map(str, [c['concept_id'] for c in exclude_concepts]))})
                )
            """)

    if not where_conditions:
        # Fallback to general EHR data presence if no specific cohort definition is given
        where_conditions.append("cb_search_person.has_ehr_data = 1") 

    final_where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

    sql_query = f"""
    SELECT
        {', '.join(select_clauses)}
    {os.linesep.join(from_join_clauses)}
    {final_where_clause}
    """
    return sql_query

def build_domain_query(domain_name: str, concepts_include: List[int], concepts_exclude: List[int], cdr_path: str, column_prefix: str = "", include_all_cols: bool = False) -> str:
    """
    Builds a SQL query for a specific domain.
    If `include_all_cols` is True for 'condition_occurrence', it selects all standard columns.
    Otherwise, it returns a binary presence or a single value (e.g., for measurement).
    """
    domain_table = f"`{cdr_path}.{domain_name}`"
    
    # Common where clauses for included/excluded concepts
    concept_filter_conditions = []
    if concepts_include:
        concept_filter_conditions.append(f"{domain_name}_concept_id IN ({','.join(map(str, concepts_include))})")
    if concepts_exclude:
        concept_filter_conditions.append(f"{domain_name}_concept_id NOT IN ({','.join(map(str, concepts_exclude))})")
    
    concept_filter_clause = ""
    if concept_filter_conditions:
        concept_filter_clause = f"WHERE {' AND '.join(concept_filter_conditions)}"

    if domain_name == 'condition_occurrence' and include_all_cols:
        
        select_cols = [
            "c_occurrence.person_id",
            "c_occurrence.condition_concept_id",
            "c_standard_concept.concept_name as standard_concept_name",
            "c_standard_concept.concept_code as standard_concept_code",
            "c_standard_concept.vocabulary_id as standard_vocabulary",
            "c_occurrence.condition_start_datetime",
            "c_occurrence.condition_end_datetime",
            "c_occurrence.condition_type_concept_id",
            "c_type.concept_name as condition_type_concept_name",
            "c_occurrence.stop_reason",
            "c_occurrence.visit_occurrence_id",
            "visit.concept_name as visit_occurrence_concept_name",
            "c_occurrence.condition_source_value",
            "c_occurrence.condition_source_concept_id",
            "c_source_concept.concept_name as source_concept_name",
            "c_source_concept.concept_code as source_concept_code",
            "c_source_concept.vocabulary_id as source_vocabulary",
            "c_occurrence.condition_status_source_value",
            "c_occurrence.condition_status_concept_id",
            "c_status.concept_name as condition_status_concept_name"
        ]
        
        from_joins = [
            f"FROM {domain_table} c_occurrence",
            f"LEFT JOIN `{cdr_path}.concept` c_standard_concept ON c_occurrence.condition_concept_id = c_standard_concept.concept_id",
            f"LEFT JOIN `{cdr_path}.concept` c_type ON c_occurrence.condition_type_concept_id = c_type.concept_id",
            f"LEFT JOIN `{cdr_path}.visit_occurrence` v ON c_occurrence.visit_occurrence_id = v.visit_occurrence_id",
            f"LEFT JOIN `{cdr_path}.concept` visit ON v.visit_concept_id = visit.concept_id",
            f"LEFT JOIN `{cdr_path}.concept` c_source_concept ON c_occurrence.condition_source_concept_id = c_source_concept.concept_id",
            f"LEFT JOIN `{cdr_path}.concept` c_status ON c_occurrence.condition_status_concept_id = c_status.concept_id"
        ]
        
        sql = f"""
        SELECT
            {', '.join(select_cols)}
        {os.linesep.join(from_joins)}
        {concept_filter_clause}
        """
        # Note: This subquery will return ALL matching conditions for ALL people.
        # The join in `load_data_from_bigquery` will filter it down to the desired cohort.
        return sql

    elif domain_name == 'measurement':
        sql = f"""
        SELECT
            m.person_id,
            m.value_as_number AS {column_prefix}value
        FROM
            `{cdr_path}.measurement` m
        WHERE
            m.measurement_concept_id IN ({','.join(map(str, concepts_include))})
            AND m.value_as_number IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY m.person_id ORDER BY m.measurement_date DESC) = 1
        """
    else: # Default for condition_occurrence (binary presence), observation (binary presence)
        sql = f"""
        SELECT
            DISTINCT t.person_id,
            1 AS {column_prefix}presence
        FROM
            {domain_table} t
        {concept_filter_clause}
        """
    return sql

def load_data_from_bigquery(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads data from Google BigQuery based on the provided configuration,
    joining multiple tables to create the final dataset.
    """
    client = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    cdr_path = get_aou_cdr_path()

    base_person_query = build_person_base_query(config)
    
    join_queries = []
    
    # Outcome
    outcome_config = config.get('outcome')
    if outcome_config:
        outcome_concepts_include = outcome_config.get('concepts_include', [])
        outcome_concepts_exclude = outcome_config.get('concepts_exclude', [])
        outcome_domain = outcome_config['domain']
        
        # Decide if you want ALL columns for the outcome (e.g., for detailed analysis)
        # or just a binary presence. For this example, let's keep it binary for joins.
        # If you wanted all columns, you'd fetch the outcome_df separately or restructure the main join.
        outcome_query = build_domain_query(outcome_domain, outcome_concepts_include, outcome_concepts_exclude, cdr_path, column_prefix=f"{outcome_config['name']}_")
        join_queries.append({
            'name': outcome_config['name'],
            'sql': outcome_query,
            'type': 'LEFT JOIN',
            'join_col_person': 'person_id',
            'join_col_subquery': 'person_id',
            'select_col': f"{outcome_config['name']}_presence" if outcome_domain != 'measurement' else f"{outcome_config['name']}_value"
        })

    # Co-indicators
    for indicator in config.get('co_indicators', []):
        indicator_concepts_include = indicator.get('concepts_include', [])
        indicator_concepts_exclude = indicator.get('concepts_exclude', [])
        indicator_domain = indicator['domain']
        indicator_query = build_domain_query(indicator_domain, indicator_concepts_include, indicator_concepts_exclude, cdr_path, column_prefix=f"{indicator['name']}_")
        join_queries.append({
            'name': indicator['name'],
            'sql': indicator_query,
            'type': 'LEFT JOIN',
            'join_col_person': 'person_id',
            'join_col_subquery': 'person_id',
            'select_col': f"{indicator['name']}_presence" if indicator_domain != 'measurement' else f"{indicator['name']}_value"
        })

    # Features (excluding person domain features which are in the base query)
    for feature in config.get('features', []):
        if feature['domain'] == 'person':
            continue
        
        feature_concepts_include = feature.get('concepts_include', [])
        feature_concepts_exclude = feature.get('concepts_exclude', [])
        feature_domain = feature['domain']
        
        if feature['name'] == 'bmi' and feature_domain == 'measurement':
             bmi_query = build_domain_query('measurement', feature_concepts_include, [], cdr_path, column_prefix=f"{feature['name']}_")
             join_queries.append({
                'name': feature['name'],
                'sql': bmi_query,
                'type': 'LEFT JOIN',
                'join_col_person': 'person_id',
                'join_col_subquery': 'person_id',
                'select_col': f"{feature['name']}_value"
            })
        elif feature_domain in ['observation', 'condition_occurrence']:
            feature_query = build_domain_query(feature_domain, feature_concepts_include, feature_concepts_exclude, cdr_path, column_prefix=f"{feature['name']}_")
            join_queries.append({
                'name': feature['name'],
                'sql': feature_query,
                'type': 'LEFT JOIN',
                'join_col_person': 'person_id',
                'join_col_subquery': 'person_id',
                'select_col': f"{feature['name']}_presence"
            })


    main_query_parts = [f"WITH base_person AS ({base_person_query})"]
    
    select_cols = ["base_person.*"]

    for i, join_info in enumerate(join_queries):
        alias = f"join_tbl_{i}"
        main_query_parts.append(f", {alias} AS ({join_info['sql']})")
        
        select_cols.append(f"{alias}.{join_info['select_col']} AS {join_info['name']}")

    final_from_clause = "FROM base_person"
    final_join_clauses = []
    for i, join_info in enumerate(join_queries):
        alias = f"join_tbl_{i}"
        final_join_clauses.append(f"{join_info['type']} {alias} ON base_person.{join_info['join_col_person']} = {alias}.{join_info['join_col_subquery']}")
        
    full_sql_query = f"""
    {os.linesep.join(main_query_parts)}
    SELECT
        {', '.join(select_cols)}
    {final_from_clause}
    {os.linesep.join(final_join_clauses)}
    """
    
    print(f"Executing BigQuery SQL query:\n{full_sql_query}")

    try:
        df = pd.read_gbq(
            full_sql_query,
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            dialect="standard",
            use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
            progress_bar_type="tqdm_notebook"
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data from BigQuery: {e}")


def filter_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Filter dataframe to retain only specified columns."""
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}. Available columns: {list(df.columns)}")
    return df[columns].copy()

def stratify_by_risk(df: pd.DataFrame, risk_column: str, threshold: float) -> pd.DataFrame:
    """Stratify dataset into high vs. low risk groups based on threshold."""
    if risk_column not in df.columns:
        raise KeyError(f"Risk column '{risk_column}' does not exist in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[risk_column]):
        try:
            df[risk_column] = pd.to_numeric(df[risk_column], errors='coerce')
            if df[risk_column].isnull().all():
                raise ValueError(f"Risk column '{risk_column}' became all NaNs after conversion. It must contain numeric data.")
            print(f"Warning: Risk column '{risk_column}' was converted to numeric dtype.")
        except Exception:
            raise ValueError(f"Risk column '{risk_column}' must contain numeric data and could not be converted.")

    df = df.copy()
    df['risk_group'] = np.where(df[risk_column] >= threshold, 'high', 'low')
    df['risk_group'] = np.where(pd.isna(df[risk_column]), 'unknown', df['risk_group'])
    return df

if __name__ == "__main__":
    config_file_path = "configuration.yaml"

    try:
        config = load_configuration(config_file_path)
        print("Configuration loaded successfully.")
        
        risk_column_name = None
        risk_threshold_value = None
        
        for feature in config.get('features', []):
            if feature['name'] == 'bmi':
                risk_column_name = 'bmi'
                # Assuming you'll add 'risk_threshold' to your BMI feature in config.yaml
                risk_threshold_value = feature.get('risk_threshold', 25.0) 
                break
        
        if not risk_column_name or risk_threshold_value is None:
            print("Warning: No explicit risk column or threshold found in configuration for stratification. Skipping stratification.")
            
        print("Loading data from BigQuery...")
        data_df = load_data_from_bigquery(config)
        print(f"Data loaded from BigQuery. Shape: {data_df.shape}")
        print("First 5 rows of data:")
        print(data_df.head())
        print("\nColumn names after loading:")
        print(data_df.columns.tolist())

        filtered_df = data_df.copy()
        print(f"\nDataFrame after initial loading. Shape: {filtered_df.shape}")

        if risk_column_name and risk_threshold_value is not None:
            if risk_column_name in filtered_df.columns:
                stratified_df = stratify_by_risk(filtered_df, risk_column_name, risk_threshold_value)
                print(f"\nData stratified by risk using '{risk_column_name}'. Unique risk groups: {stratified_df['risk_group'].unique()}")
                print("Counts per risk group:")
                print(stratified_df['risk_group'].value_counts())
                print("\nStratified data head:")
                print(stratified_df.head())
            else:
                print(f"Warning: Risk column '{risk_column_name}' not found in the loaded DataFrame. Skipping stratification.")
        else:
            print("Skipping stratification as risk column or threshold not fully specified in config.")

    except RuntimeError as e:
        print(f"A data loading or configuration error occurred: {e}")
    except KeyError as e:
        print(f"A column or key error occurred: {e}. This might mean a concept was not found or mapped incorrectly.")
    except EnvironmentError as e:
        print(f"Environment setup error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
