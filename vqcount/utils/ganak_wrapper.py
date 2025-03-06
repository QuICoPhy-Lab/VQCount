"""
Python wrapper for the Ganak solver to count the number of solutions of a SAT problem.
"""

import subprocess
import tempfile


def ganak_counter(cnf_file_path):
    """
    Python Wrapper for the model counter Ganak to count the number of solutions of a SAT problem (see https://github.com/meelgroup/ganak).

    Parameters
    ----------
    cnf_file_path : str
        Path to the CNF formula file.

    Returns
    -------
    numsol : int
        Number of solutions.
    """

    # Creating a temporary file to store the output
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:

        # CHANGE THE BUILD PATH HERE!

        # Run the ganak command and redirect the output to the temporary file
        sub = subprocess.run(
            ["ganak/build/ganak", "-delta", "0.0001", "-noPMC", cnf_file_path],
            text=True,
            stdout=temp_file,
            check=False,
        )

        # Move the file pointer to the beginning of the file
        temp_file.seek(0)

        # Extract the number of solutions from the output
        numsol = None
        for line in temp_file:
            if line.startswith("s mc"):
                numsol = int(line.split()[2])
                break

    return numsol
