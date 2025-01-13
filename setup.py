from setuptools import setup, find_packages

setup(
    name='fish-classifier',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.10.1',
        'streamlit',
        'pillow==9.4.0',
        'requests==1.22.4'
    ],
    include_package_data=True,
    description='A fish disease classifier using transfer learning.',
    author='Ronit Bhowmick',
    author_email='your-email@example.com',
    url='https://github.com/Ronit-Bhowmick/Fish-Disease-Classifier.git',
)
