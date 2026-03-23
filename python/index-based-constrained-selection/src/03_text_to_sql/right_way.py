"""
RIGHT WAY: Indexed query template selection.

This module demonstrates the secure approach to Text-to-SQL:
- LLM selects from pre-validated query templates by index
- System validates all parameters before rendering
- Impossible to generate invalid or dangerous SQL
- Guaranteed type safety and schema compliance

✅ RECOMMENDED: Use this approach for all production Text-to-SQL systems.
"""

from typing import Any
from dataclasses import dataclass

from ..common.registry import IndexedRegistry
from ..common.models import QuerySelection
from .schema import (
    QUERY_TEMPLATES,
    get_template_registry,
    validate_parameters,
    render_template,
    get_schema_description,
)


@dataclass
class TemplateSelectionResult:
    """Result of template selection and rendering."""
    template_index: int
    template_name: str
    parameters: dict[str, Any]
    rendered_query: str
    is_valid: bool
    validation_message: str


class TemplateSQLGenerator:
    """
    Secure SQL generator using indexed template selection.
    
    This class demonstrates the safe approach: the LLM only selects from
    pre-validated templates by index. The system handles parameter validation
    and template rendering, making SQL injection and syntax errors impossible.
    
    Example:
        >>> generator = TemplateSQLGenerator()
        >>> # LLM returns: template_index=1, parameters={"user_id": "123"}
        >>> result = generator.generate(1, {"user_id": "123"})
        >>> print(result.rendered_query)
        SELECT * FROM orders WHERE user_id = 123 ORDER BY created_at DESC
        >>> print(result.is_valid)
        True
    """
    
    def __init__(self):
        """Initialize the template-based SQL generator."""
        # Create indexed registry of query templates
        self.registry = IndexedRegistry[str](
            items={
                idx: f"[{idx}] {template['name']}: {template['description']}"
                for idx, template in QUERY_TEMPLATES.items()
            }
        )
        self.schema_description = get_schema_description()
    
    def get_available_templates(self) -> str:
        """
        Get formatted list of available templates for LLM context.
        
        Returns:
            Multi-line string describing all available query templates.
        """
        lines = [
            "Available Query Templates:",
            "==========================",
            "",
            "Select the template index that best matches the user's request.",
            "",
        ]
        
        for idx, template in QUERY_TEMPLATES.items():
            lines.append(f"[{idx}] {template['name']}")
            lines.append(f"    Description: {template['description']}")
            lines.append(f"    Parameters: {', '.join(template['params'])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate(
        self,
        template_index: int,
        parameters: dict[str, Any]
    ) -> TemplateSelectionResult:
        """
        Generate SQL by rendering a pre-validated template.
        
        Args:
            template_index: Index of the template to use (from registry)
            parameters: Dictionary of parameter names to values
            
        Returns:
            TemplateSelectionResult with rendered query and validation info
        """
        # Validate template index exists
        if template_index not in QUERY_TEMPLATES:
            return TemplateSelectionResult(
                template_index=template_index,
                template_name="INVALID",
                parameters=parameters,
                rendered_query="",
                is_valid=False,
                validation_message=f"Invalid template index: {template_index}"
            )
        
        template = QUERY_TEMPLATES[template_index]
        
        # Validate parameters match template requirements
        is_valid, validation_message = validate_parameters(template_index, parameters)
        
        if not is_valid:
            return TemplateSelectionResult(
                template_index=template_index,
                template_name=template["name"],
                parameters=parameters,
                rendered_query="",
                is_valid=False,
                validation_message=validation_message
            )
        
        # Render the template with validated parameters
        try:
            rendered_query = render_template(template_index, parameters)
            
            return TemplateSelectionResult(
                template_index=template_index,
                template_name=template["name"],
                parameters=parameters,
                rendered_query=rendered_query,
                is_valid=True,
                validation_message="Query rendered successfully"
            )
        except ValueError as e:
            return TemplateSelectionResult(
                template_index=template_index,
                template_name=template["name"],
                parameters=parameters,
                rendered_query="",
                is_valid=False,
                validation_message=str(e)
            )
    
    def simulate_llm_selection(self, user_request: str) -> QuerySelection:
        """
        Simulate an LLM selecting the appropriate template for a request.
        
        In production, this would be an actual LLM call. Here we simulate
        the expected behavior to demonstrate the pattern.
        
        Args:
            user_request: Natural language description of desired query
            
        Returns:
            QuerySelection with template index and parameters
        """
        request_lower = user_request.lower()
        
        # Simulate LLM reasoning and selection
        # In reality, the LLM would see the template registry and decide
        
        if "user" in request_lower and ("id" in request_lower or any(c.isdigit() for c in user_request)):
            # Extract user ID if present
            user_id = "".join(filter(str.isdigit, user_request)) or "123"
            
            if "order" in request_lower:
                if "count" in request_lower or "how many" in request_lower:
                    return QuerySelection(
                        template_index=2,  # get_order_count
                        parameters={"user_id": user_id},
                        reasoning="Request asks for count of orders for a user"
                    )
                else:
                    return QuerySelection(
                        template_index=1,  # get_user_orders
                        parameters={"user_id": user_id},
                        reasoning="Request asks for orders belonging to a user"
                    )
            else:
                return QuerySelection(
                    template_index=0,  # get_user_by_id
                    parameters={"user_id": user_id},
                    reasoning="Request asks for user information by ID"
                )
        
        if "recent" in request_lower or "latest" in request_lower:
            return QuerySelection(
                template_index=3,  # get_recent_orders
                parameters={
                    "date": "2024-01-01",
                    "limit": "10"
                },
                reasoning="Request asks for recent orders with date filtering"
            )
        
        if "email" in request_lower:
            return QuerySelection(
                template_index=4,  # get_user_by_email
                parameters={"email": "user@example.com"},
                reasoning="Request asks to find user by email address"
            )
        
        if "high value" in request_lower or "expensive" in request_lower:
            return QuerySelection(
                template_index=5,  # get_high_value_orders
                parameters={"min_total": "1000"},
                reasoning="Request asks for high-value orders above a threshold"
            )
        
        # Default fallback
        return QuerySelection(
            template_index=1,  # get_user_orders
            parameters={"user_id": "123"},
            reasoning="Default fallback: get orders for user"
        )
    
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for template selection.
        
        This is much simpler than free-form generation prompts because
        the LLM only needs to select, not generate.
        """
        return f"""You are a query selector. Your job is to select the most appropriate 
pre-defined query template based on the user's request.

{self.get_available_templates()}

Instructions:
1. Read the user's request carefully
2. Select the template index that best matches their intent
3. Extract the required parameters from their request
4. Return ONLY a JSON object with:
   - template_index: integer
   - parameters: object with parameter names and values

You CANNOT generate arbitrary SQL. You MUST select from the available templates.
"""


def demonstrate_safety() -> None:
    """
    Demonstrate the safety and reliability of template-based SQL generation.
    
    This function shows how the indexed approach prevents all the failure
    modes demonstrated in the wrong_way module.
    """
    generator = TemplateSQLGenerator()
    
    print("=" * 70)
    print("RIGHT WAY: Indexed Template Selection")
    print("=" * 70)
    print("\nDemonstrating guaranteed safety...\n")
    
    # Show the available templates
    print(generator.get_available_templates())
    
    test_requests = [
        "Find all orders for user 123",
        "Get count of orders for user 456",
        "Get recent orders from 2024-01-01, limit to 10",
        "Find user with email user@example.com",
        "Get high value orders above $1000",
    ]
    
    print("\n" + "=" * 70)
    print("Template Selection and Rendering")
    print("=" * 70)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{'─' * 70}")
        print(f"Request {i}: {request}")
        print("─" * 70)
        
        # Simulate LLM selecting template
        selection = generator.simulate_llm_selection(request)
        print(f"LLM Selection:")
        print(f"  Template Index: {selection.template_index}")
        print(f"  Parameters: {selection.parameters}")
        
        # Generate SQL from selection
        result = generator.generate(selection.template_index, selection.parameters)
        
        print(f"\nGenerated SQL:\n  {result.rendered_query}")
        print(f"\nValidation: {'✓ PASSED' if result.is_valid else '✗ FAILED'}")
        print(f"Message: {result.validation_message}")
    
    # Demonstrate attack resistance
    print(f"\n{'=' * 70}")
    print("ATTACK RESISTANCE DEMONSTRATION")
    print("=" * 70)
    
    attack_attempts = [
        ("SQL injection attempt", 0, {"user_id": "123'; DROP TABLE users; --"}),
        ("Invalid template index", 99, {"user_id": "123"}),
        ("Missing parameters", 0, {}),
        ("Extra parameters (injection)", 0, {"user_id": "123", "malicious": "DROP TABLE users"}),
    ]
    
    for attack_name, template_idx, params in attack_attempts:
        print(f"\n{'─' * 70}")
        print(f"Attack: {attack_name}")
        print(f"  Template Index: {template_idx}")
        print(f"  Parameters: {params}")
        print("─" * 70)
        
        result = generator.generate(template_idx, params)
        
        if result.is_valid:
            print(f"⚠️  UNEXPECTED: Query was accepted!")
            print(f"   SQL: {result.rendered_query}")
        else:
            print(f"✓ BLOCKED: {result.validation_message}")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY: Template selection is inherently safe")
    print("=" * 70)
    print("""
Security guarantees:
1. ✓ SQL injection impossible - only template parameters are substituted
2. ✓ Invalid table names impossible - templates are pre-validated
3. ✓ Syntax errors impossible - templates are syntactically correct
4. ✓ Destructive operations impossible - no DROP/DELETE templates exist
5. ✓ Consistent output format - always valid SQL
6. ✓ Simple prompt engineering - just select from list

The LLM cannot generate dangerous SQL because it can only select templates.
The system validates all parameters before rendering.
""")


if __name__ == "__main__":
    demonstrate_safety()
