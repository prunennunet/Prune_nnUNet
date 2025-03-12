


def format_scientific(num):
    # Convert to scientific notation with 0 decimal places
    formatted = f"{num:.0e}"
    # Remove leading zeros in the exponent part
    parts = formatted.split('e')
    exponent = parts[1].lstrip('+-0')
    # Rebuild with correct sign
    if parts[1].startswith('-'):
        return f"{parts[0]}e-{exponent}"
    else:
        return f"{parts[0]}e+{exponent}"
