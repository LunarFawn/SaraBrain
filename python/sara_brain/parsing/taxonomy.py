"""Bootstrap taxonomy: maps known properties to their category.

This is what allows the brain to create meaningful intermediate relation neurons.
When taught "apples are red", the taxonomy tells us "red" is a color, so
the intermediate neuron becomes "fruit_color" (subject_category + property_type).
"""

from __future__ import annotations


# property_label → property_type
_PROPERTY_MAP: dict[str, str] = {
    # Colors
    "red": "color",
    "blue": "color",
    "green": "color",
    "yellow": "color",
    "orange": "color",
    "purple": "color",
    "black": "color",
    "white": "color",
    "brown": "color",
    "pink": "color",
    "crimson": "color",
    # Shapes
    "round": "shape",
    "square": "shape",
    "triangular": "shape",
    "flat": "shape",
    "oval": "shape",
    "cylindrical": "shape",
    "spherical": "shape",
    # Tastes
    "sweet": "taste",
    "sour": "taste",
    "bitter": "taste",
    "salty": "taste",
    "savory": "taste",
    "spicy": "taste",
    # Textures
    "smooth": "texture",
    "rough": "texture",
    "soft": "texture",
    "hard": "texture",
    "fuzzy": "texture",
    "crunchy": "texture",
    # Sizes
    "big": "size",
    "small": "size",
    "large": "size",
    "tiny": "size",
    "huge": "size",
    # Temperatures
    "hot": "temperature",
    "cold": "temperature",
    "warm": "temperature",
    "cool": "temperature",
}

# subject_label → subject_category
_CATEGORY_MAP: dict[str, str] = {
    "apple": "fruit",
    "banana": "fruit",
    "orange": "fruit",
    "grape": "fruit",
    "strawberry": "fruit",
    "lemon": "fruit",
    "cherry": "fruit",
    "mango": "fruit",
    "pear": "fruit",
    "peach": "fruit",
    "circle": "geometric",
    "square": "geometric",
    "triangle": "geometric",
    "rectangle": "geometric",
    "sphere": "geometric",
    "cube": "geometric",
    "dog": "animal",
    "cat": "animal",
    "bird": "animal",
    "fish": "animal",
    "horse": "animal",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "firetruck": "vehicle",
}


class Taxonomy:
    """Extensible taxonomy for property types and subject categories."""

    def __init__(self) -> None:
        self._properties: dict[str, str] = dict(_PROPERTY_MAP)
        self._categories: dict[str, str] = dict(_CATEGORY_MAP)

    def property_type(self, label: str) -> str:
        """Return the property type for a label, or 'attribute' as default."""
        return self._properties.get(label.lower(), "attribute")

    def subject_category(self, label: str) -> str:
        """Return the category for a subject, or 'thing' as default."""
        return self._categories.get(label.lower(), "thing")

    def relation_label(self, subject: str, property_label: str) -> str:
        """Generate the intermediate relation neuron label.

        E.g., subject="apple", property="red" → "apple_color"
        """
        prop_type = self.property_type(property_label)
        return f"{subject}_{prop_type}"

    def register_property(self, label: str, prop_type: str) -> None:
        self._properties[label.lower()] = prop_type

    def register_category(self, label: str, category: str) -> None:
        self._categories[label.lower()] = category
