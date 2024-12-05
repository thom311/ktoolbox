from ktoolbox import k8sClient


def test_create() -> None:
    assert k8sClient.K8sClient is not None
