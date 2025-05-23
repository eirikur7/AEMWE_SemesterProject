import os
import json
import numpy as np
import pandas as pd
from itertools import product
from pyDOE2 import lhs, fracfact
from scipy.spatial.distance import cdist
from scipy.stats import qmc


class DOEGenerator:
    def __init__(
        self,
        parameters_filename: str,
        parameters_path: str = "data_DOE_input_parameters",
        export_path: str = "data_DOE_output_results",
        KOH_concentration: float = None,
    ):
        """
        Initialize the DOEGenerator with a parameter definition JSON file.

        Args:
            parameters_filename (str): Filename of the parameter JSON.
            parameters_path (str): Path to the input parameter files.
            export_path (str): Path to save generated DOE samples.
            KOH_concentration (float, optional): Used for computing linked values.
        """
        self.parameter_file = parameters_filename
        self.export_path = export_path
        self.prefix = os.path.splitext(parameters_filename)[0]
        self.KOH_conc = KOH_concentration
        self.param_defs = self._load_parameters(
            os.path.join(parameters_path, parameters_filename)
        )
        self.param_values = self._get_scaled_values()
        self.param_names = list(self.param_values.keys())  # Only include parameters with actual values

    def _load_parameters(self, filepath: str) -> dict:
        """Load parameter definitions from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get("parameters", {})

    def _get_scaled_values(self) -> dict:
        """
        Apply scaling and offsets to raw parameter values.

        Returns:
            dict: Scaled values per parameter.
        """
        scaled = {}
        for name, meta in self.param_defs.items():
            vals = meta.get("values")
            if vals is None:
                continue  # Skip parameters without explicit values
            scale = meta.get("scaling_factor", 1.0)
            offset = meta.get("offset", 0.0)
            scaled[name] = [(v * scale + offset) for v in vals]
        return scaled

    def _add_linked_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional parameters that are functionally dependent on existing ones.

        If a known input value is matched, a predefined value is used.
        Otherwise, a mathematical expression is used (e.g., for temperature-based exchange currents).

        Args:
            df (pd.DataFrame): The dataframe of sampled parameters.

        Returns:
            pd.DataFrame: Dataframe with linked parameters added.
        """
        tol = 1e-8
        for base, meta in self.param_defs.items():
            linked = meta.get("linked")
            if not linked:
                continue

            raw_vals = meta.get("values", [])
            use_formula = False

            if base == "T":
                temps = df[base].values
                for v in temps:
                    if not any(abs(v - rv) < tol for rv in raw_vals):
                        use_formula = True
                        break
                if use_formula:
                    if self.KOH_conc is None:
                        raise ValueError("KOH concentration required for linked math formulas")
                    logK = np.log(self.KOH_conc)

            for link_name, link_list in linked.items():
                def compute(v):
                    for rv, lv in zip(raw_vals, link_list):
                        if abs(v - rv) < tol:
                            return lv
                    if use_formula:
                        if link_name == 'i0_ref_H2':
                            return (1.18 * logK + 6.27) * np.exp(-1758 / (v + 273.15))
                        elif link_name == 'i0_ref_O2':
                            return (0.53 * logK + 335) * np.exp(-1458 / (v + 273.15))
                    raise KeyError(f"No linked formula for parameter {link_name} at value {v}")

                df[link_name] = df[base].map(compute)

        return df

    def _save(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Save the DOE dataframe to CSV with units as string suffixes."""
        filename = f"{self.prefix}_DOE_{method}.csv"
        os.makedirs(self.export_path, exist_ok=True)
        out = df.copy()
        for col in out.columns:
            meta = self.param_defs.get(col, {})
            unit = meta.get("convert_to") or meta.get("unit")
            if unit:
                out[col] = out[col].map(lambda x: f"{x:.6g}[{unit}]")
        out.to_csv(os.path.join(self.export_path, filename), index=False)
        print(f"Saved: {filename}")
        return out

    def _export_to_comsol_txt(self, df: pd.DataFrame, method: str):
        """Export DOE data in COMSOL-readable plain text format."""
        lines = []
        for p in df.columns:
            vals = df[p].round(6).tolist()
            vals_str = ", ".join(f"{v:.6g}" for v in vals)
            meta = self.param_defs.get(p, {})
            unit = meta.get("convert_to") or meta.get("unit", "")
            lines.append(f'{p} "{vals_str}" [{unit}]')
        fn = f"{self.prefix}_DOE_{method}.txt"
        os.makedirs(self.export_path, exist_ok=True)
        with open(os.path.join(self.export_path, fn), 'w') as f:
            f.write("\n".join(lines))
        print(f"Exported COMSOL-formatted file: {fn}")

    def latin_hypercube(self, samples: int = 20, KOH_concentration: float = None):
        """
        Generate a Latin Hypercube Sample (LHS) of the parameter space.

        Ensures stratified sampling along each parameter dimension,
        preventing clustering and capturing the full range of input values.
        """
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        n = len(self.param_names)
        mat = lhs(n, samples=samples)
        bounds = [self.param_values[nm] for nm in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])
        scaled = mat * (maxs - mins) + mins
        df = pd.DataFrame(scaled, columns=self.param_names)
        df = self._add_linked_parameters(df)
        self._save(df, 'latin_hypercube')
        self._export_to_comsol_txt(df, 'latin_hypercube')

    def maximin_latin_hypercube(self, samples: int = 20, iterations: int = 50, KOH_concentration: float = None):
        """
        Generate a Maximin-enhanced Latin Hypercube Sample.

        Iteratively generates LHS designs and selects the one that
        maximizes the minimum distance between points — improving
        space-filling properties.
        """
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        n = len(self.param_names)
        best_dist = -np.inf
        best_sample = None
        bounds = [self.param_values[nm] for nm in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])

        for _ in range(iterations):
            sample = lhs(n, samples=samples)
            scaled = sample * (maxs - mins) + mins
            dists = cdist(scaled, scaled)
            np.fill_diagonal(dists, np.inf)
            min_dist = dists.min()
            if min_dist > best_dist:
                best_dist = min_dist
                best_sample = scaled

        df = pd.DataFrame(best_sample, columns=self.param_names)
        df = self._add_linked_parameters(df)
        self._save(df, 'maximin_lhs')
        self._export_to_comsol_txt(df, 'maximin_lhs')

    def sobol_sampling(self, samples: int = 20, KOH_concentration: float = None):
        """
        Generate samples using a Sobol sequence.

        Sobol is a quasi-random low-discrepancy sequence that offers
        better uniformity and reproducibility than random or LHS.
        Ideal for surrogate modeling with limited sample budgets.
        """
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        n = len(self.param_names)
        sampler = qmc.Sobol(d=n, scramble=True)
        raw = sampler.random(n=samples)
        bounds = [self.param_values[nm] for nm in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])
        scaled = raw * (maxs - mins) + mins
        df = pd.DataFrame(scaled, columns=self.param_names)
        df = self._add_linked_parameters(df)
        self._save(df, 'sobol')
        self._export_to_comsol_txt(df, 'sobol')


    def full_factorial(self, KOH_concentration: float = None):
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        grids = [self.param_values[n] for n in self.param_names]
        df = pd.DataFrame(list(product(*grids)), columns=self.param_names)
        df = self._add_linked_parameters(df)
        self._save(df, 'full_factorial')
        self._export_to_comsol_txt(df, 'full_factorial')

    def random_sampling(self, samples: int = 20, KOH_concentration: float = None):
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        rng = np.random.default_rng()
        bounds = [self.param_values[nm] for nm in self.param_names]
        mins = np.array([min(v) for v in bounds])
        maxs = np.array([max(v) for v in bounds])
        df = pd.DataFrame(rng.uniform(mins, maxs, (samples, len(self.param_names))),
                          columns=self.param_names)
        df = self._add_linked_parameters(df)
        return self._save(df, 'random_sampling')

    def grid_sampling(self, levels_per_param: int = 3, KOH_concentration: float = None):
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        grids = [np.linspace(min(v), max(v), levels_per_param) for v in self.param_values.values()]
        df = pd.DataFrame(list(product(*grids)), columns=self.param_names)
        df = self._add_linked_parameters(df)
        return self._save(df, 'grid_sampling')

    def fractional_factorial(self, base_design: str = "a b c ab ac bc abc", KOH_concentration: float = None):
        if KOH_concentration is not None:
            self.KOH_conc = KOH_concentration
        design = base_design.replace(' ', '')
        if len(design) > 7:
            raise ValueError("Maximum 7 base factors supported.")
        subs = self.param_names[:len(design)]
        bounds = [self.param_values[nm] for nm in subs]
        data = fracfact(base_design)
        if data.shape[1] != len(bounds):
            raise ValueError("Mismatch between design and number of parameters.")
        scaled = np.zeros_like(data)
        for i, (low, high) in enumerate((min(v), max(v)) for v in bounds):
            scaled[:, i] = ((data[:, i] + 1) / 2) * (high - low) + low
        df = pd.DataFrame(scaled, columns=subs)
        df = self._add_linked_parameters(df)
        return self._save(df, 'fractional_factorial')
