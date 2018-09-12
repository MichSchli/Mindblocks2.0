from Mindblocks.controller.parameter_searcher.search_configuration import SearchConfiguration


class ParameterSearcher:

    variable_repository = None
    ml_helper_factory = None

    def __init__(self, variable_repository, ml_helper_factory, logger_manager):
        self.variable_repository = variable_repository
        self.ml_helper_factory = ml_helper_factory
        self.logger_manager = logger_manager

    def count_search_options(self, greedy):
        count_mode = "train"
        return self.variable_repository.count_search_options(mode=count_mode, greedy=greedy)

    def greedy_search(self, graph, minimize_valid_score):
        search_option_count = self.count_search_options(True)
        self.logger_manager.log("Starting greedy search over " + str(search_option_count) + " configurations...", "search", "status")
        search_options = self.build_greedy_search_options()
        best_search_option = self.find_optimal(graph, search_options, minimize_valid_score)

        return best_search_option

    def grid_search(self, graph, minimize_valid_score):
        search_option_count = self.count_search_options(False)
        self.logger_manager.log("Starting grid search over " + str(search_option_count) + " configurations...", "search", "status")
        search_options = self.build_grid_search_options()
        best_search_option = self.find_optimal(graph, search_options, minimize_valid_score)

        return best_search_option

    def build_grid_search_options(self):
        count_mode = "train"
        true_default = SearchConfiguration()
        search_configs = [true_default]

        for variable in self.variable_repository.get_all():
            search_options = variable.count_search_options(mode=count_mode)
            for idx, search_option in enumerate(search_options):
                for search_config in search_configs[:]:
                    search_config.register(variable.get_name(), idx, 0)

                    for alt in range(1,search_option):
                        search_config_copy = search_config.copy()
                        search_config_copy.register(variable.get_name(), idx, alt)
                        search_configs.append(search_config_copy)

        return search_configs

    def build_greedy_search_options(self):
        count_mode = "train"
        true_default = SearchConfiguration()
        search_configs = [true_default]

        for variable in self.variable_repository.get_all():
            search_options = variable.count_search_options(mode=count_mode)
            for idx, search_option in enumerate(search_options):
                for search_config in search_configs:
                    search_config.register(variable.get_name(), idx, 0)

                for alt in range(1,search_option):
                    default_copy = true_default.copy()
                    default_copy.register(variable.get_name(), idx, alt)
                    search_configs.append(default_copy)

        return search_configs

    def find_optimal(self, graph, search_configurations, minimize_valid_score):
        best_score = None
        best_configuration = None
        best_idx = 0
        best_epoch = 0
        for idx, search_configuration in enumerate(search_configurations):
            self.variable_repository.apply_search_configuration(search_configuration)
            ml_helper = self.ml_helper_factory.build_ml_helper_from_graph(graph, minimize_valid_score=minimize_valid_score)
            ml_helper.initialize_model()
            ml_helper.train()
            cfg_validation_score = ml_helper.get_best_validation_score()
            epoch = ml_helper.get_best_epoch()

            self.logger_manager.log(
                "Configuration #" + str(idx) + " finished with validation score " + str(
                    cfg_validation_score) + " at epoch " + str(epoch) + ":", "search", "status")

            if best_score is None \
                    or ((best_score > cfg_validation_score and minimize_valid_score)
                        or (best_score < cfg_validation_score and not minimize_valid_score)):
                self.logger_manager.log("This is the running optimum...",
                                        "search", "status")

                best_score = cfg_validation_score
                best_configuration = search_configuration
                best_idx = idx
                best_epoch = epoch

        self.logger_manager.log("Search complete. Optimal configuration was  #" + str(best_idx) + " with validation score " + str(best_score) + " at epoch " + str(best_epoch) + ":", "search", "status")
        self.variable_repository.describe_search_configuration(search_configuration)


        return best_configuration