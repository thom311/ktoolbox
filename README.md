ktoolbox
========

This is a utility library with various Python helpers. Use or don't.

It works with Python 3.9 or newer and implements typing hints.

The API is not guaranteed to be stable. However, an effort shall be made that
every API change leads to an mypy warning for affected users. Install via `pip
install git+https://github.com/thom311/ktoolbox@$VERSION`. As the API is not
stable, you are advised to refer to a specific commit sha as version.


## Attribution

This is based on code by Balazs Nemeth, William Zhao and Salvatore Daniele and
the projects [ocp-traffic-flow-tests](https://github.com/wizhaoredhat/ocp-traffic-flow-tests) and
[cluster-deployment-automation](https://github.com/bn222/cluster-deployment-automation).


## Wheel Files

You can find compiled wheels files at https://thom311.github.io/ktoolbox/wheels/ .

You can install via:

  - `pip install git+https://github.com/thom311/ktoolbox@$VERSION`
  - `pip install https://thom311.github.io/ktoolbox/wheels/$WHEEL`
  - `pip install --find-links=https://thom311.github.io/ktoolbox/wheels 'ktoolbox>=0.8'`

The benefit of using wheel files is that it's cheaper to install and you can
copy the wheel file around.

The benefit of "git+https://" URL is that it can contain the full git SHA
sum to cryptographically ensure we install what was intended.
