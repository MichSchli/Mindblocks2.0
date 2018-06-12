class ContractionComponentTypeModel:

    name = "__contraction__"

    def execute(self, in_sockets, value, language="python"):
        c1_in_sockets = len(value.source_component.in_sockets)
        c1_output = value.source_component.component_type.execute(in_sockets, value.source_component.value, language=language)

        c2_input = value.distribute_to_target(c1_output, in_sockets[c1_in_sockets:])
        c2_output = value.target_component.component_type.execute(c2_input, value.target_component.value, language=language)

        output = c2_output + value.distribute_to_output(c1_output)

        return output