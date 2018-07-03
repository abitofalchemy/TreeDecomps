  333  virtualenv --python=python3 --system-site-packages orange3venv
  335  del -rf orange3venv/
  336  /bin/rm -rf orange3venv/
  343  brew install pyenv-virtualenvwrapper
  348  pyenv install 3.6.0
  349  pyenv install 2.7.13
  350  pyenv virtualenv 3.6.0 jupyter3
  351  pyenv virtualenv 2.7.13 ipython2
  353  pyenv activate jupyter3
  354  brew install pyenv-virtualenv
  355  brew install pyenv
  356  pyenv activate jupyter3
  365  pyenv virtualenv 3.6.0 jupyter3
  367  pyenv activate orange3
  372  pyenv activate jupyter3
  376  pyenv deactivate
  381  pyenv activate ipython2
  382  pyenv virtualenv 2.7.13 ipython2
  383  pyenv activate ipython2
  386  pyenv deactivate
  387  pyenv global 3.6.0 2.7.13 jupyter3 ipython2
  388  pyenv which python
  389  pyenv which python2
  390  pyenv which jupyter
  391  pyenv which ipython
  392  pyenv which ipython2
  517  conda info --envs
  531  cd .pyenv/
  544  history | grep env
  545  history | grep env > ../Docs/pyenv.md
