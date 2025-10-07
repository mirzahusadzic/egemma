from src.server import SentenceTransformerWrapper


def test_get_optimal_device_cuda_available(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert SentenceTransformerWrapper._get_optimal_device() == "cuda"


def test_get_optimal_device_mps_available(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert SentenceTransformerWrapper._get_optimal_device() == "mps"


def test_get_optimal_device_cpu_only(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    assert SentenceTransformerWrapper._get_optimal_device() == "cpu"


def test_get_optimal_device_cuda_preferred_over_mps(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    assert SentenceTransformerWrapper._get_optimal_device() == "cuda"
