"""
Database schema and query template definitions for Text-to-SQL example.

This module defines a simple database schema and pre-validated query templates
to demonstrate the benefits of constrained query selection over free-form SQL generation.
"""

from typing import TypedDict, List, Dict, Any


class TableSchema(TypedDict):
    """Type definition for table schema."""
    name: str
    columns: List[str]


class QueryTemplate(TypedDict):
    """Type definition for query template."""
    name: str
    template: str
    params: List[str]
    description: str


# Database schema definitions
USERS_TABLE: TableSchema = {
    "name": "users",
    "columns": ["id", "name", "email", "created_at"]
}

ORDERS_TABLE: TableSchema = {
    "name": "orders",
    "columns": ["id", "user_id", "total", "status", "created_at"]
}

# Pre-validated query templates
# These templates are validated at build time and cannot be modified by the LLM
QUERY_TEMPLATES: Dict[int, QueryTemplate] = {
    0: {
        "name": "get_user_by_id",
        "template": "SELECT * FROM users WHERE id = {user_id}",
        "params": ["user_id"],
        "description": "Get a single user by their ID"
    },
    1: {
        "name": "get_user_orders",
        "template": "SELECT * FROM orders WHERE user_id = {user_id} ORDER BY created_at DESC",
        "params": ["user_id"],
        "description": "Get all orders for a user"
    },
    2: {
        "name": "get_order_count",
        "template": "SELECT COUNT(*) as count FROM orders WHERE user_id = {user_id}",
        "params": ["user_id"],
        "description": "Count orders for a user"
    },
    3: {
        "name": "get_recent_orders",
        "template": "SELECT * FROM orders WHERE created_at > '{date}' ORDER BY created_at DESC LIMIT {limit}",
        "params": ["date", "limit"],
        "description": "Get recent orders with date filter"
    },
    4: {
        "name": "get_user_by_email",
        "template": "SELECT * FROM users WHERE email = '{email}'",
        "params": ["email"],
        "description": "Get user by email address"
    },
    5: {
        "name": "get_high_value_orders",
        "template": "SELECT * FROM orders WHERE total > {min_total} ORDER BY total DESC",
        "params": ["min_total"],
        "description": "Get orders above a minimum total"
    },
}


def get_schema_description() -> str:
    """
    Generate a human-readable description of the database schema.
    
    Returns:
        String describing all tables and their columns.
    """
    lines = ["Database Schema:", "================"]
    
    for table in [USERS_TABLE, ORDERS_TABLE]:
        lines.append(f"\nTable: {table['name']}")
        lines.append(f"  Columns: {', '.join(table['columns'])}")
    
    return "\n".join(lines)


def get_template_registry() -> Dict[int, str]:
    """
    Create a registry mapping template indices to descriptions.
    
    This is used to present available queries to the LLM for selection.
    
    Returns:
        Dictionary mapping template index to description string.
    """
    return {
        idx: f"[{idx}] {template['name']}: {template['description']}"
        for idx, template in QUERY_TEMPLATES.items()
    }


def validate_parameters(template_index: int, parameters: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that provided parameters match the template requirements.
    
    Args:
        template_index: Index of the query template to validate against
        parameters: Dictionary of parameter names to values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if template_index not in QUERY_TEMPLATES:
        return False, f"Invalid template index: {template_index}"
    
    template = QUERY_TEMPLATES[template_index]
    required_params = set(template["params"])
    provided_params = set(parameters.keys())
    
    # Check for missing parameters
    missing = required_params - provided_params
    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}"
    
    # Check for extra parameters (potential injection attempt)
    extra = provided_params - required_params
    if extra:
        return False, f"Unexpected parameters provided: {', '.join(extra)}"
    
    return True, ""


def sanitize_parameter(value: Any, param_type: str = "string") -> str:
    """
    Sanitize a parameter value to prevent SQL injection.
    
    This is a demonstration sanitizer. In production, use your database driver's
    parameterized query support instead of string substitution.
    
    Args:
        value: The parameter value to sanitize
        param_type: Type of parameter ("string", "numeric", "date")
        
    Returns:
        Sanitized parameter value safe for SQL insertion
        
    Raises:
        ValueError: If parameter contains dangerous content
    """
    value_str = str(value)
    
    # Block SQL keywords and dangerous characters
    dangerous_patterns = [
        ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE",
        "UNION", "SELECT", "FROM", "WHERE", "OR", "AND", "EXEC", "EXECUTE"
    ]
    
    upper_value = value_str.upper()
    for pattern in dangerous_patterns:
        if pattern in upper_value:
            raise ValueError(f"Parameter contains dangerous content: '{pattern}'")
    
    # For numeric parameters, ensure they are actually numeric
    if param_type == "numeric":
        try:
            float(value_str)
        except ValueError:
            raise ValueError(f"Numeric parameter contains non-numeric characters: {value_str}")
    
    # Escape single quotes for string parameters
    if param_type == "string":
        value_str = value_str.replace("'", "''")
    
    return value_str


def render_template(template_index: int, parameters: Dict[str, Any]) -> str:
    """
    Render a query template with the provided parameters.
    
    Args:
        template_index: Index of the query template to render
        parameters: Dictionary of parameter names to values
        
    Returns:
        Rendered SQL query string
        
    Raises:
        ValueError: If parameters are invalid or template doesn't exist
    """
    is_valid, error = validate_parameters(template_index, parameters)
    if not is_valid:
        raise ValueError(error)
    
    template = QUERY_TEMPLATES[template_index]["template"]
    template_def = QUERY_TEMPLATES[template_index]
    
    # Determine parameter types based on template
    # In production, use proper parameterized queries with your database driver
    param_types = {}
    for param in template_def["params"]:
        if param in ["user_id", "min_total", "limit"]:
            param_types[param] = "numeric"
        elif param in ["date"]:
            param_types[param] = "date"
        else:
            param_types[param] = "string"
    
    # Sanitize and substitute parameters
    for param_name, param_value in parameters.items():
        try:
            sanitized = sanitize_parameter(param_value, param_types.get(param_name, "string"))
            template = template.replace(f"{{{param_name}}}", sanitized)
        except ValueError as e:
            raise ValueError(f"Parameter '{param_name}' validation failed: {e}")
    
    return template
