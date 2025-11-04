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


## Installation and Wheel Files

You can find pre-compiled wheels files at https://thom311.github.io/ktoolbox/wheels/ and on
the [gh-pages](https://github.com/thom311/ktoolbox/commits/gh-pages/) branch.

You can install via:

  - `pip install --force-reinstall git+https://github.com/thom311/ktoolbox@$VERSION`
  - `pip install --force-reinstall https://thom311.github.io/ktoolbox/wheels/$WHEEL`
  - `pip install --force-reinstall --no-index --find-links=https://thom311.github.io/ktoolbox/wheels 'ktoolbox>=0.8'`

The benefit of using wheel files is that it's cheaper to install and you can
copy the wheel file around.

The benefit of "git+https://" URL is that it can contain the full git SHA
sum to cryptographically ensure what is installed.

Alternatively, you can also use (e.g.)

  - `pip install --force-reinstall https://raw.githubusercontent.com/thom311/ktoolbox/cb38f23eb271cbbe38ea83357132362eade61c49/wheels/140.7e3728001e18/ktoolbox-0.8.0-py3-none-any.whl`

which has the benefit that the git commit sha cryptographically refers to the
content of a known commit on the `gh-pages` branch. You still have no guarantee
that the content of that wheel file is something in particular. But you have the
guarantee that this is served from a known commit on the `gh-pages` branch.

Environment Variables
=====================

ktoolbox honors some environment variables. Find out which via
`git grep -w getenv_config`.

Variables:
  - `KTOOLBOX_CURRENT_HOST` (overwrites `ktoolbox.common.get_current_host()`).

Logging related for loggers configured by `ktoolbox.common.log_config_logger()`:
  - `KTOOLBOX_ALL_LOGGERS` (enable logging for all python loggers)
  - `KTOOLBOX_LOGFILE` (write log output also to file, including a level)
  - `KTOOLBOX_LOGLEVEL` (overwrite the default level)
  - `KTOOLBOX_LOGSTDOUT` (log to stdout instead of stderr)
  - `KTOOLBOX_LOGTAG` (add a tag/prefix to each message)

Other:
  - `KUBECONFIG` (fallback for `ktoolbox.k8sClient.K8sClient()`)
  - `USER` (fallback ssh user login for `ktoolbox.host.RemoteHost()`)
