import importlib
import inspect
import re


def get_pruning_function(class_name):
    """
    Get the apply function for a given pruning method class.

    Args:
        class_name (str): The name of the pruning method class (without 'Method' suffix)

    Returns:
        function: The apply function for the pruning method
    """
    # Convert CamelCase to snake_case (e.g., "RangePruning" -> "range_pruning")
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()

    # Replace "pruning" with "prune" for the file name (e.g., "range_pruning" -> "range_prune")
    if "pruning" in snake_case:
        file_name = snake_case.replace("pruning", "prune")
    else:
        file_name = snake_case

    # Construct the module path
    module_path = f"prune_nnunet.prune_method.{file_name}"

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Find the function that starts with "apply_"
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and name.startswith("apply_"):
                return obj

        raise ValueError(f"No 'apply_' function found in {module_path}")

    except ImportError:
        raise ImportError(f"Could not import module {module_path}")
