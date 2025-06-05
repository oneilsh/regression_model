import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from google.cloud import bigquery
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)

# Mock environment variables for local testing if not in AoU Workbench
# os.environ["WORKSPACE_CDR"] = "all-of-us-research-workbench-####.r2023q3_unzipped_data"
# os.environ["GOOGLE_CLOUD_PROJECT"] = "your-gcp-project-id"

# --- Keep existing functions (load_configuration, get_aou_cdr_path, build_cohort_criteria_sql, build_person_base_query) ---
# (Place these above build_domain_query if you are replacing everything from build_domain_query downwards)

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
        if concept_id is None:
            raise ValueError(f"Concept in cohort_definition must have 'concept_id': {concept_entry}")

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
        exclude_person_sql = build_cohort_criteria_sql(exclude_concepts, cdr_path, operator="IN")

        if include_sql:
            where_conditions.append(f"({include_sql})")
        if exclude_person_sql:
            exclude_concept_ids = [c['concept_id'] for c in exclude_concepts if 'concept_id' in c]
            if exclude_concept_ids:
                where_conditions.append(f"""
                    person.person_id NOT IN (
                        SELECT DISTINCT csae_excl.person_id
                        FROM `{cdr_path}.cb_search_all_events` csae_excl
                        WHERE csae_excl.concept_id IN ({','.join(map(str, exclude_concept_ids))})
                    )
                """)

    if not where_conditions:
        where_conditions.append("cb_search_person.has_ehr_data = 1")

    final_where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

    sql_query = f"""
    SELECT
        {', '.join(select_clauses)}
    {os.linesep.join(from_join_clauses)}
    {final_where_clause}
    """
    return sql_query


# --- build_domain_query function ---
def build_domain_query(domain_name: str, concepts_include: List[int], concepts_exclude: List[int], cdr_path: str, column_prefix: str = "", include_all_cols: bool = False) -> str:
    """
    Builds a SQL query for a specific domain.
    If `include_all_cols` is True for 'condition_occurrence', it selects all standard columns.
    Otherwise, it returns a binary presence or a single value (e.g., for measurement).
    """
    sql = "" # Initialize sql to an empty string
    
    domain_table_name = domain_name
    domain_table = f"`{cdr_path}.{domain_table_name}`"

    # Determine the correct concept ID column name for the domain
    concept_id_col_name = "" # Initialize
    if domain_name == 'condition_occurrence':
        concept_id_col_name = 'condition_concept_id'
    elif domain_name == 'observation':
        concept_id_col_name = 'observation_concept_id'
    elif domain_name == 'measurement':
        concept_id_col_name = 'measurement_concept_id'
    elif domain_name == 'drug_exposure':
        concept_id_col_name = 'drug_concept_id'
    elif domain_name == 'procedure_occurrence':
        concept_id_col_name = 'procedure_concept_id'
    elif domain_name == 'ds_survey':
        concept_id_col_name = 'question_concept_id'

    # Common where clauses for included/excluded concepts
    concept_filter_conditions = []

    # Handle concepts_include
    if concepts_include:
        concepts_str_include = ','.join(map(str, concepts_include))
        if domain_name == 'ds_survey':
            concept_filter_conditions.append(f"t.question_concept_id IN ({concepts_str_include})")
        elif concept_id_col_name:
            concept_filter_conditions.append(f"t.{concept_id_col_name} IN ({concepts_str_include})")
        else:
            logging.warning(f"No specific concept_id column determined for domain '{domain_name}' for inclusion. "
                            "Concepts will not be filtered by ID for this domain.")
    else: # If concepts_include is empty, generate a condition that matches nothing
        if domain_name not in ['person']:
            concept_filter_conditions.append("FALSE")

    # Handle concepts_exclude
    if concepts_exclude:
        concepts_str_exclude = ','.join(map(str, concepts_exclude))
        if domain_name == 'ds_survey':
            concept_filter_conditions.append(f"t.question_concept_id NOT IN ({concepts_str_exclude})")
        elif concept_id_col_name:
            concept_filter_conditions.append(f"t.{concept_id_col_name} NOT IN ({concepts_str_exclude})")
        else:
            logging.warning(f"No specific concept_id column determined for domain '{domain_name}' for exclusion. "
                            "Concepts will not be filtered by ID for this domain (exclusion).")

    concept_filter_clause = ""
    if concept_filter_conditions:
        concept_filter_clause = f"WHERE {' AND '.join(concept_filter_conditions)}"


    # --- Assign SQL query based on domain and options ---
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
    else: # Default for binary presence for most domains including observation, ds_survey, drug_exposure, procedure_occurrence
        sql = f"""
        SELECT
            DISTINCT t.person_id,
            1 AS {column_prefix}presence
        FROM
            {domain_table} t
        {concept_filter_clause}
        """
    return sql

# --- build_observation_duration_query function ---
def build_observation_duration_query(cdr_path: str, column_prefix: str = "", concepts_include: Optional[List[int]] = None) -> str:
    """
    Builds a SQL query to calculate a patient's total observation duration in days
    from the OMOP observation_period table.
    Note: The 'concepts_include' parameter is ignored for this general observation period calculation
          as observation_period is not concept-specific.
    """
    if concepts_include: # Log a warning if concepts are passed but not used for this query
        logging.warning("Concepts are provided for 'observation_duration' feature, "
                        "but 'observation_period' table is used which is not concept-specific. "
                        "Concepts will be ignored for this query.")
    
    sql = f"""
    SELECT
        person_id,
        DATE_DIFF(MAX(observation_period_end_date), MIN(observation_period_start_date), DAY) AS {column_prefix}duration_days
    FROM
        `{cdr_path}.observation_period`
    WHERE
        observation_period_start_date IS NOT NULL AND observation_period_end_date IS NOT NULL
        AND DATE_DIFF(observation_period_end_date, observation_period_start_date, DAY) >= 0
    GROUP BY
        person_id
    """
    return sql

# --- build_condition_datetime_query function ---
def build_condition_datetime_query(cdr_path: str, concepts_include: List[int], column_prefix: str = "") -> str:
    """
    Builds a SQL query to get the min condition_start_date and max condition_end_date
    for a given set of condition concepts for each person.
    """
    if not concepts_include:
        # If no concepts are provided, this query will return nothing useful,
        # so return a query that produces an empty table with the correct schema
        return f"""
        SELECT
            person_id,
            CAST(NULL AS DATE) AS {column_prefix}min_start_date,
            CAST(NULL AS DATE) AS {column_prefix}max_end_date
        FROM `{cdr_path}.person`
        WHERE FALSE
        """

    concepts_str = ','.join(map(str, concepts_include))

    sql = f"""
    SELECT
        person_id,
        MIN(condition_start_date) AS {column_prefix}min_start_date,
        MAX(condition_end_date) AS {column_prefix}max_end_date
    FROM
        `{cdr_path}.condition_occurrence`
    WHERE
        condition_concept_id IN ({concepts_str})
        AND condition_start_date IS NOT NULL
    GROUP BY
        person_id
    """
    return sql


# --- load_data_from_bigquery function ---
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
        
        # Determine the select_col alias for the outcome
        if outcome_domain == 'measurement':
            select_col_alias = f"{outcome_config['name']}_value"
        else:
            select_col_alias = f"{outcome_config['name']}_presence"
        
        outcome_query = build_domain_query(outcome_domain, outcome_concepts_include, outcome_concepts_exclude, cdr_path, column_prefix=f"{outcome_config['name']}_")
        join_queries.append({
            'name': outcome_config['name'],
            'sql': outcome_query,
            'type': 'LEFT JOIN',
            'join_col_person': 'person_id',
            'join_col_subquery': 'person_id',
            'select_col': select_col_alias
        })

    # Co-indicators
    for indicator in config.get('co_indicators', []):
        indicator_concepts_include = indicator.get('concepts_include', [])
        indicator_concepts_exclude = indicator.get('concepts_exclude', [])
        indicator_domain = indicator['domain']
        indicator_name = indicator['name']

        # Determine the select_col alias for indicators
        if indicator_domain == 'measurement':
            select_col_alias = f"{indicator_name}_value"
        else:
            select_col_alias = f"{indicator_name}_presence"

        indicator_query = build_domain_query(indicator_domain, indicator_concepts_include, indicator_concepts_exclude, cdr_path, column_prefix=f"{indicator_name}_")
        join_queries.append({
            'name': indicator_name,
            'sql': indicator_query,
            'type': 'LEFT JOIN',
            'join_col_person': 'person_id',
            'join_col_subquery': 'person_id',
            'select_col': select_col_alias
        })

    # Features (excluding person domain features which are in the base query)
    # Maintain a set of features that have already been handled (e.g., if a pair is processed together)
    handled_features = set()

    for feature in config.get('features', []):
        feature_name = feature['name']

        # Skip if this feature was already handled as part of a pair
        if feature_name in handled_features:
            continue

        # Handle 'observation_duration' feature specifically
        if feature_name == 'observation_duration':
            # Get the COPD concepts from the outcome config (for filtering condition_occurrence based duration)
            outcome_config = config.get('outcome', {})
            copd_concepts_include = outcome_config.get('concepts_include', [])

            feature_query = build_observation_duration_query(
                cdr_path,
                column_prefix=f"{feature_name}_",
                concepts_include=copd_concepts_include # Pass COPD concepts here, though they will be ignored by observation_period logic
            )
            join_queries.append({
                'name': feature_name,
                'sql': feature_query,
                'type': 'LEFT JOIN',
                'join_col_person': 'person_id',
                'join_col_subquery': 'person_id',
                'select_col': f"{feature_name}_duration_days"
            })
            handled_features.add(feature_name) # Mark as handled
            continue

        # Handle paired condition start/end datetimes (condition_start_datetimes, condition_end_datetimes)
        elif feature_name == 'condition_start_datetimes' or feature_name == 'condition_end_datetimes':
            # This block will be triggered by either name, but ensure the query is built only once
            
            # Get the COPD concepts from the outcome config for these specific dates
            outcome_config = config.get('outcome', {})
            copd_concepts_include = outcome_config.get('concepts_include', [])

            # Build the query for min/max dates ONCE
            # Use a consistent subquery prefix for clarity in the generated SQL
            temp_query_prefix = "copd_condition_datetimes_"
            condition_datetimes_query = build_condition_datetime_query(
                cdr_path,
                concepts_include=copd_concepts_include,
                column_prefix=temp_query_prefix
            )
            
            # Add join_query for min_start_date
            join_queries.append({
                'name': 'condition_start_datetimes', # This is the final column name
                'sql': condition_datetimes_query,
                'type': 'LEFT JOIN',
                'join_col_person': 'person_id',
                'join_col_subquery': 'person_id',
                'select_col': f"{temp_query_prefix}min_start_date" # Selects the column from the subquery
            })
            handled_features.add('condition_start_datetimes') # Mark as handled

            # Add join_query for max_end_date (using the same underlying query)
            join_queries.append({
                'name': 'condition_end_datetimes', # This is the final column name
                'sql': condition_datetimes_query,
                'type': 'LEFT JOIN',
                'join_col_person': 'person_id',
                'join_col_subquery': 'person_id',
                'select_col': f"{temp_query_prefix}max_end_date" # Selects the column from the subquery
            })
            handled_features.add('condition_end_datetimes') # Mark as handled

            continue # Both date features are handled, skip to next feature in config


        # Handle other 'person' domain features that are directly in base_person_query
        if feature['domain'] == 'person':
            continue # These are already in the base_person_query, no separate join needed

        # Handle all other generic features (measurement, observation, ds_survey, condition_occurrence etc.)
        feature_concepts_include = feature.get('concepts_include', [])
        feature_concepts_exclude = feature.get('concepts_exclude', [])
        feature_domain = feature['domain']

        # Determine the select_col alias based on domain type
        if feature_domain == 'measurement':
            select_col_alias = f"{feature_name}_value"
        else: # Covers observation, condition_occurrence, ds_survey, drug_exposure, procedure_occurrence, etc.
            select_col_alias = f"{feature_name}_presence"

        feature_query = build_domain_query(feature_domain, feature_concepts_include, feature_concepts_exclude, cdr_path, column_prefix=f"{feature_name}_")
        join_queries.append({
            'name': feature_name,
            'sql': feature_query,
            'type': 'LEFT JOIN',
            'join_col_person': 'person_id',
            'join_col_subquery': 'person_id',
            'select_col': select_col_alias
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
    # This block is for direct testing of the dataloader.py script
    # It assumes environment variables are set and a config.yaml exists at root.
    config_file_path = "configuration.yaml"

    try:
        config = load_configuration(config_file_path)
        print("Configuration loaded successfully.")
        
        # This part of __main__ would need updates if used directly for stratification etc.
        # It's primarily for testing the data loading function.
        
        print("Loading data from BigQuery...")
        data_df = load_data_from_bigquery(config)
        print(f"Data loaded from BigQuery. Shape: {data_df.shape}")
        print("First 5 rows of data:")
        print(data_df.head())
        print("\nColumn names after loading:")
        print(data_df.columns.tolist())

    except RuntimeError as e:
        print(f"A data loading or configuration error occurred: {e}")
    except KeyError as e:
        print(f"A column or key error occurred: {e}. This might mean a concept was not found or mapped incorrectly.")
    except EnvironmentError as e:
        print(f"Environment setup error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
