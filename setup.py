from setuptools import setup, find_packages

setup(
    name='wee_agent',
    version='0.1.2',
    description='A micro agent for open_ai llm model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Haifeng Kong',
    author_email="redknox@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    exclude_package_data={"": ["tests", "agents", "decorators", "script"]},
    install_requires=[
        'jsonschema == 4.21.1',
        'openai == 1.29.0',
        'pydantic == 2.6.4',
        'python-dotenv == 1.0.1',
        'tiktoken == 0.7.0',
        'genson==1.2.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',

)
