from setuptools import find_packages,setup
from typing import List


HIPHEN_e="-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    Docstring for get_requirements
    
    :param filepath: Description
    :type filepath: str
    :return: Description
    :rtype: List[str]
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HIPHEN_e in requirements:
            requirements.remove(HIPHEN_e)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author="Deepak",
    author_email="deepakambati5@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)