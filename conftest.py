def pytest_addoption(parser):
    parser.addoption('--audit', action='store_true', default=False, dest='audit', help="Enables user interface for checking and modifying test results.")
