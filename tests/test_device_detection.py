from src.embedding import _get_optimal_device


def test_get_optimal_device_force_cpu(mocker):
    """Test that FORCE_CPU setting overrides hardware detection."""
    from src.config import settings

    mocker.patch.object(settings, "FORCE_CPU", True)
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert _get_optimal_device() == "cpu"


def test_get_optimal_device_cuda_available(mocker):
    from src.config import settings

    mocker.patch.object(settings, "FORCE_CPU", False)
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert _get_optimal_device() == "cuda"


def test_get_optimal_device_mps_available(mocker):
    from src.config import settings

    mocker.patch.object(settings, "FORCE_CPU", False)
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert _get_optimal_device() == "mps"


def test_get_optimal_device_cpu_only(mocker):
    from src.config import settings

    mocker.patch.object(settings, "FORCE_CPU", False)
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert _get_optimal_device() == "cpu"


def test_get_optimal_device_cuda_preferred_over_mps(mocker):
    from src.config import settings

    mocker.patch.object(settings, "FORCE_CPU", False)
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert _get_optimal_device() == "cuda"
