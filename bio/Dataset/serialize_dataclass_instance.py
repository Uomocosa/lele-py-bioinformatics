import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

def serialize_dataclass_instance(instance):
    """Dynamically serializes the dataclass, ignoring callables."""
    def _serialize(obj):
        if dataclasses.is_dataclass(obj):
            # Iterate through fields, skip if the value is a callable
            return {
                f.name: _serialize(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if not callable(getattr(obj, f.name))
            }
        elif isinstance(obj, Path):
            # Use .as_posix() instead of str() for cross-platform JSON safety
            return obj.as_posix()
        elif isinstance(obj, (list, tuple)):
            # Reconstruct the list or tuple with serialized items
            return type(obj)(_serialize(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj
        
    return _serialize(instance)


def test_():
    # 1. Define a dummy callable to test exclusion
    def dummy_handler(data):
        return data

    # 2. Define a nested dataclass
    @dataclass
    class Options:
        n_points: int = 2
        save_path: Path = Path("/usr/local/data")

    # 3. Define the main dataclass mixing types
    @dataclass
    class Config:
        name: str = "MLP_Config"
        percentages: tuple = (0.6, 0.2, 0.2)
        options: Options = field(default_factory=Options)
        handle_data: Callable = dummy_handler  # This should be ignored
        
    # Instantiate the config
    config = Config()
    
    # Run the serializer
    serialized_data = serialize_dataclass_instance(config)
    
    # Print the output to verify
    print("--- Original Dataclass ---")
    print(config)
    print("\n--- Serialized Dictionary ---")
    print(serialized_data)
    
    # 4. Assertions to guarantee it worked
    assert "handle_data" not in serialized_data, "Failed: Callable was not ignored!"
    assert serialized_data["name"] == "MLP_Config", "Failed: String serialization mismatch."
    assert serialized_data["percentages"] == (0.6, 0.2, 0.2), "Failed: Tuple serialization mismatch."
    assert isinstance(serialized_data["options"]["save_path"], str), "Failed: Path was not converted to string."
    # Now this will safely pass on Windows too!
    assert serialized_data["options"]["save_path"] == "/usr/local/data", "Failed: Path string value mismatch."
    
    print("\nAll tests passed successfully! The dictionary is ready for JSON saving.")
