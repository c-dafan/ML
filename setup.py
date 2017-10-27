from distutils.core import setup

setup(name='machinelearn',
        version='1.0',
        package_dir={'machinelearn': ''},
        packages=['machinelearn','machinelearn.score','machinelearn.base',\
                  'machinelearn.Dimension','machinelearn.bayers','machinelearn.cluster', \
                  'machinelearn.knn','machinelearn.regression','machinelearn.svm',\
                  'machinelearn.tree'],
)