"""
Test data generator for different function categories and sample sizes.

Generates test data for:
- Oceanographic domain functions (SA, CT, p, lat)
- Ice functions (t_ice, p_ice)
- Salinity conversions (SP)
- Profile functions (multi-dimensional arrays)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# Sample sizes to generate
SAMPLE_SIZES = [10, 100, 1000, 10000]


def generate_oceanographic_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate oceanographic domain test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SA, CT, p, lat arrays
    """
    return {
        "SA": np.linspace(30.0, 40.0, n, dtype=np.float64),  # Absolute Salinity, g/kg
        "CT": np.linspace(0.0, 25.0, n, dtype=np.float64),   # Conservative Temperature, °C
        "p": np.linspace(0.0, 5000.0, n, dtype=np.float64),  # Pressure, dbar
        "lat": np.linspace(-60.0, 60.0, n, dtype=np.float64),  # Latitude, degrees
    }


def generate_ice_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate ice function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with t_ice, p_ice arrays
    """
    return {
        "t_ice": np.linspace(-50.0, 0.0, n, dtype=np.float64),  # Ice temperature, °C
        "p_ice": np.linspace(0.0, 1000.0, n, dtype=np.float64),  # Ice pressure, dbar
    }


def generate_salinity_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate salinity conversion test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SP, SA arrays
    """
    return {
        "SP": np.linspace(0.0, 42.0, n, dtype=np.float64),  # Practical Salinity
        "SA": np.linspace(30.0, 40.0, n, dtype=np.float64),  # Absolute Salinity, g/kg
    }


def generate_temperature_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate temperature conversion test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with t, t68, CT, pt arrays
    """
    return {
        "t": np.linspace(-2.0, 40.0, n, dtype=np.float64),  # In-situ temperature, °C
        "t68": np.linspace(-2.0, 40.0, n, dtype=np.float64),  # IPTS-68 temperature, °C
        "CT": np.linspace(0.0, 25.0, n, dtype=np.float64),  # Conservative Temperature, °C
        "pt": np.linspace(0.0, 25.0, n, dtype=np.float64),  # Potential temperature, °C
    }


def generate_profile_data(n: int, n_profiles: int = 5) -> Dict[str, np.ndarray]:
    """
    Generate profile function test data (multi-dimensional).
    
    Parameters
    ----------
    n : int
        Number of depth levels per profile
    n_profiles : int
        Number of profiles
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with profile arrays
    """
    # Create 2D arrays: (n_profiles, n_levels)
    SA = np.tile(np.linspace(30.0, 40.0, n), (n_profiles, 1)).astype(np.float64)
    CT = np.tile(np.linspace(0.0, 25.0, n), (n_profiles, 1)).astype(np.float64)
    p = np.tile(np.linspace(0.0, 5000.0, n), (n_profiles, 1)).astype(np.float64)
    lat = np.tile(np.linspace(-60.0, 60.0, n_profiles), (n, 1)).T.astype(np.float64)
    
    return {
        "SA": SA,
        "CT": CT,
        "p": p,
        "lat": lat,
    }


def generate_freezing_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate freezing point function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SA, p, saturation_fraction arrays
    """
    return {
        "SA": np.linspace(0.0, 42.0, n, dtype=np.float64),  # Absolute Salinity, g/kg
        "p": np.linspace(0.0, 10000.0, n, dtype=np.float64),  # Pressure, dbar
        "saturation_fraction": np.linspace(0.0, 1.0, n, dtype=np.float64),  # Saturation fraction
    }


def generate_enthalpy_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate enthalpy function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SA, CT, p, h arrays
    """
    return {
        "SA": np.linspace(30.0, 40.0, n, dtype=np.float64),
        "CT": np.linspace(0.0, 25.0, n, dtype=np.float64),
        "p": np.linspace(0.0, 5000.0, n, dtype=np.float64),
        "h": np.linspace(0.0, 150000.0, n, dtype=np.float64),  # Enthalpy, J/kg
    }


def generate_entropy_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate entropy function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SA, CT, entropy arrays
    """
    return {
        "SA": np.linspace(30.0, 40.0, n, dtype=np.float64),
        "CT": np.linspace(0.0, 25.0, n, dtype=np.float64),
        "entropy": np.linspace(3000.0, 5000.0, n, dtype=np.float64),  # Entropy, J/(kg·K)
    }


def generate_density_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate density function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with SA, CT, p, rho arrays
    """
    return {
        "SA": np.linspace(30.0, 40.0, n, dtype=np.float64),
        "CT": np.linspace(0.0, 25.0, n, dtype=np.float64),
        "p": np.linspace(0.0, 5000.0, n, dtype=np.float64),
        "rho": np.linspace(1020.0, 1080.0, n, dtype=np.float64),  # Density, kg/m³
    }


def generate_geostrophy_data(n: int) -> Dict[str, np.ndarray]:
    """
    Generate geostrophy function test data.
    
    Parameters
    ----------
    n : int
        Number of samples
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with lon, lat, p, geo_strf arrays
    """
    return {
        "lon": np.linspace(-180.0, 180.0, n, dtype=np.float64),  # Longitude, degrees
        "lat": np.linspace(-60.0, 60.0, n, dtype=np.float64),  # Latitude, degrees
        "p": np.linspace(0.0, 5000.0, n, dtype=np.float64),
        "geo_strf": np.linspace(-100.0, 100.0, n, dtype=np.float64),  # Geostrophic streamfunction
    }


def generate_all_test_data():
    """
    Generate all test data for different categories and sample sizes.
    
    Returns
    -------
    Dict[str, Dict[int, Dict[str, np.ndarray]]]
        Nested dictionary: category -> sample_size -> data_dict
    """
    all_data = {}
    
    categories = {
        "oceanographic": generate_oceanographic_data,
        "ice": generate_ice_data,
        "salinity": generate_salinity_data,
        "temperature": generate_temperature_data,
        "freezing": generate_freezing_data,
        "enthalpy": generate_enthalpy_data,
        "entropy": generate_entropy_data,
        "density": generate_density_data,
        "geostrophy": generate_geostrophy_data,
    }
    
    for category, generator_func in categories.items():
        all_data[category] = {}
        for size in SAMPLE_SIZES:
            all_data[category][size] = generator_func(size)
    
    # Generate profile data separately (different structure)
    all_data["profile"] = {}
    for size in SAMPLE_SIZES:
        # Use smaller profile sizes for large n
        n_profiles = 5 if size <= 100 else 3
        all_data["profile"][size] = generate_profile_data(size, n_profiles)
    
    return all_data


def save_test_data(output_dir: Path):
    """
    Generate and save all test data to NPZ files.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save test data files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating test data...")
    all_data = generate_all_test_data()
    
    # Save each category
    for category, data_by_size in all_data.items():
        # Save each sample size
        for size, data_dict in data_by_size.items():
            filename = output_dir / f"{category}_{size}.npz"
            np.savez_compressed(filename, **data_dict)
            print(f"Saved {filename}")
    
    # Also save a combined file with metadata
    metadata = {
        "sample_sizes": SAMPLE_SIZES,
        "categories": list(all_data.keys()),
    }
    
    metadata_file = output_dir / "metadata.json"
    import json
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to {metadata_file}")
    print(f"\nTotal files created: {sum(len(data_by_size) for data_by_size in all_data.values())}")


def main():
    """Main function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "tests" / "test_data"
    
    save_test_data(output_dir)
    print(f"\nTest data generation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
