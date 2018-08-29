class MlHelperConfiguration:

    validate_every_n = None
    report_loss_every_n = None
    max_iterations = None

    report_perplexity = {"train": False,
                         "validate:": False,
                         "test": False}