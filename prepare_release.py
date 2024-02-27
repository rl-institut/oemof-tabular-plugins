import shutil
import os

# Remove the folders generated by `python setup.py sdist bdist_wheel`
for dist_file in ("dist", "build", "src/*.egg-info"):
    shutil.rmtree(dist_file, ignore_errors=True)

# Delete the previous package_data folder
#pkg_data_folder = os.path.join("src", "multi_vector_simulator", "package_data")
#shutil.rmtree(pkg_data_folder, ignore_errors=True)

# Reconstitute the package_data folder by copying the content of
# `input_template` into `src/multi-vector-simulator/package_data/input_template`,
# `report/asset` into `src/multi-vector-simulator/package_data/assets` and
# `tests/inputs` into `src/multi-vector-simulator/package_data/inputs`

# for pgk_data_src, pkg_data_dest in zip(
#     ("report/assets", "tests/inputs", "input_template"),
#     ("assets", "inputs", "input_template"),
# ):
#     shutil.copytree(
#         os.path.join(".", pgk_data_src), os.path.join(pkg_data_folder, pkg_data_dest)
#     )

# Rebuild the package
os.system("python -m build")

# Check the package
os.system("twine check dist/*")
