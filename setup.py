from setuptools import setup, find_packages

setup(
        name='forrest',
        version='0.1',
        scripts=['plotting.py', 'encoding.py', 'audio_preprocessing.py', 'preprocessing.py', 'subspace.py'],
        packages=find_packages())
