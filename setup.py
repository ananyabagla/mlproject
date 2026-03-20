from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
    # Filter out editable installs and empty lines
    requirements = [req for req in requirements if req and not req.startswith('-e')]
    return requirements
   

setup(
    name='mlops_project',
    version='0.0.1',
    author='Ananya Bagla',
    author_email = 'ananyabagla584@gmail.com',
    description='A machine learning project for MLOps',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)