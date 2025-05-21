import os
import json
import numpy as np
import pandas as pd
from itertools import product
from pyDOE2 import lhs
from sklearn.preprocessing import MinMaxScaler

class DOEGenerator:
    def __init__(self, parameters_filename: str, parameters_path: str = "data_DOE_input_parameters", export_path: str = "data_DOE_output_results"):
        """
        Initializes the DOE generator.
        """
        self.parameter_file = parameters_filename
        self.export_path = export_path
        self.prefix = os.path.splitext(parameters_filename)[0]
        self.param_defs = self._load_parameters(os.path.join(parameters_path, parameters_filename))
        self.param_names = list(self.param_defs.keys())
        self.param_values = self._get_scaled_values()

    def _load_parameters(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data["parameters"]

    def _get_scaled_values(self):
        """
        Applies scaling and returns value lists for all parameters.
        """
        scaled_values = {}
        for name, meta in self.param_defs.items():
            vals = meta["values"]
            scale = meta.get("scaling_factor", 1.0)
            offset = meta.get("offset", 0.0)
            scaled = [(v * scale + offset) for v in vals]
            scaled_values[name] = scaled
        return scaled_values

    def _add_linked_parameters(self, df: pd.DataFrame):
        """
        Adds linked parameters (e.g., i0_ref_H2, i0_ref_O2) based on values of T (assumed in °C).
        """
        if "T" in self.param_defs and "linked" in self.param_defs["T"]:
            T_raw = self.param_defs["T"]["values"]
            linked = self.param_defs["T"]["linked"]

            def get_linked_val(temp_val, link_vals):
                try:
                    # Match exact float
                    idx = T_raw.index(round(temp_val, 2))
                    return link_vals[idx]
                except ValueError:
                    raise ValueError(f"T = {temp_val} not found in original T list: {T_raw}")

            for linked_name, linked_vals in linked.items():
                df[linked_name] = df["T"].map(lambda v: get_linked_val(v, linked_vals))

        return df


    def _save(self, df: pd.DataFrame, method_name: str):
        filename = f"{self.prefix}_DOE_{method_name}.csv"
        os.makedirs(self.export_path, exist_ok=True)
        df.to_csv(os.path.join(self.export_path, filename), index=False)
        print(f"Saved: {filename}")
        return df
    
    def _export_to_comsol_txt(self, df: pd.DataFrame, method_name: str):
        """
        Exports DOE to a .txt file in COMSOL format.
        Each parameter is a row: param "val1, val2, ..." [unit]
        """
        output_lines = []
        for param in df.columns:
            values = df[param].round(6).tolist()
            values_str = ", ".join(f"{v:.6g}" for v in values)

            # Quote formatting
            if param.startswith("i0_ref_"):
                line = f'{param} {values_str} [A/m^2]'
            else:
                meta = self.param_defs.get(param, {})
                unit = meta.get("convert_to") or meta.get("unit", "")
                line = f'{param} "{values_str}" [{unit}]'

            output_lines.append(line)

        filename = f"{self.prefix}_DOE_{method_name}.txt"
        full_path = os.path.join(self.export_path, filename)
        os.makedirs(self.export_path, exist_ok=True)
        with open(full_path, "w") as f:
            f.write("\n".join(output_lines))

        print(f"Exported COMSOL-formatted file: {filename}")


    def full_factorial(self):
        grids = [self.param_values[name] for name in self.param_names]
        design = list(product(*grids))
        df = pd.DataFrame(design, columns=self.param_names)
        df = self._add_linked_parameters(df)
        
        
        self._save(df, "full_factorial")
        
        self._export_to_comsol_txt(df, "full_factorial")
        self._save(df, "full_factorial")

    def latin_hypercube(self, samples: int = 20, KOH_concentration: float = 1000.0):
        """
        Generates a Latin Hypercube design. Samples values between the min and max of each parameter.
        Calculates J0_an and J0_cath using provided [KOH] concentration (mol/m^3).
        """
        import numpy as np
        from pyDOE2 import lhs

        num_params = len(self.param_names)
        lhs_samples = lhs(num_params, samples=samples)

        # Get bounds (min and max) from values
        bounds = [self.param_values[name] for name in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])

        # Scale LHS into real-world ranges
        scaled = lhs_samples * (maxs - mins) + mins
        df = pd.DataFrame(scaled, columns=self.param_names)

        # Calculate exchange current densities if T is present
        if "T" in df.columns:
            T_vals = df["T"].values  # assumed to be in °C
            T_K = T_vals + 273.15    # convert to Kelvin

            logKOH = np.log(KOH_concentration)
            df["i0_ref_H2"] = (1.18 * logKOH + 6.27) * np.exp(-1758 / T_K)
            df["i0_ref_O2"] = (0.53 * logKOH + 335) * np.exp(-1458 / T_K)

        self._save(df, "latin_hypercube")
        
        self._export_to_comsol_txt(df, "latin_hypercube")
        self._save(df, "latin_hypercube")


    def random_sampling(self, samples: int = 20):
        rng = np.random.default_rng()
        bounds = [self.param_values[name] for name in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])
        randoms = rng.uniform(mins, maxs, size=(samples, len(self.param_names)))
        df = pd.DataFrame(randoms, columns=self.param_names)
        df = self._add_linked_parameters(df)
        return self._save(df, "random_sampling")

    def grid_sampling(self, levels_per_param: int = 3):
        grids = [
            np.linspace(min(v), max(v), levels_per_param)
            for v in self.param_values.values()
        ]
        design = list(product(*grids))
        df = pd.DataFrame(design, columns=self.param_names)
        df = self._add_linked_parameters(df)
        return self._save(df, "grid_sampling")

    def fractional_factorial(self, base_design: str = "a b c ab ac bc abc"):
        # Use only the first N parameters corresponding to letters used
        from pyDOE2 import fracfact

        clean_design = base_design.replace(" ", "")
        if len(clean_design) > 7:
            raise ValueError("Maximum 7 base factors supported.")

        subset_names = self.param_names[:len(clean_design)]
        bounds = [self.param_values[name] for name in subset_names]
        data = fracfact(base_design)
        if data.shape[1] != len(bounds):
            raise ValueError("Mismatch between design and number of parameters.")

        scaled = np.zeros_like(data)
        for i, (low, high) in enumerate((min(v), max(v)) for v in bounds):
            scaled[:, i] = ((data[:, i] + 1) / 2) * (high - low) + low

        df = pd.DataFrame(scaled, columns=subset_names)
        df_full = pd.concat([df], axis=1)
        df_full = self._add_linked_parameters(df_full)
        return self._save(df_full, "fractional_factorial")
