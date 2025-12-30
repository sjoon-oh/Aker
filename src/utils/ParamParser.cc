// 
// Author: Sukjoon Oh

#include "utils/ParamParser.hh"

namespace topkache
{
    ParameterParser::ParameterParser(std::string path)
        : file_path(path)
    {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(file_path, pt);

        std::stringstream ss;

        // Get all the keys and values from the JSON file
        for (const auto& item : pt)
        {
            // ss << item.first << ": " << item.second.get_value<std::string>() << "\n";

            parameter_map[item.first] = item.second.get_value<std::string>();
        }

        parameter.vector_dim            = std::stoi(parameter_map["vector_dim"]);
        parameter.vector_pool_size      = std::stoi(parameter_map["vector_pool_size"]);
        parameter.vector_list_size      = std::stoi(parameter_map["vector_list_size"]);
        parameter.vector_data_size      = std::stoi(parameter_map["vector_data_size"]);
        parameter.vector_intopk         = std::stoi(parameter_map["vector_intopk"]);
        parameter.vector_extras         = std::stoi(parameter_map["vector_extras"]);

        int similar_match               = std::stoi(parameter_map["similar_match"]);
        if (similar_match == 0)
        {
            parameter.similar_match     = false;
            parameter.fixed_threshold   = 0.0f;
        }
        else
        {
            parameter.similar_match     = true;
            parameter.fixed_threshold   = std::stof(parameter_map["fixed_threshold"]);

            if (parameter.fixed_threshold > 0.0f)
                parameter.use_fixed_threshold = true;
            else
                parameter.use_fixed_threshold = false;
        }

        parameter.start_threshold       = std::stof(parameter_map["start_threshold"]);
        parameter.risk_threshold        = std::stof(parameter_map["risk_threshold"]);

        parameter.alpha_tighten         = std::stof(parameter_map["alpha_tighten"]);
        parameter.alpha_loosen          = std::stof(parameter_map["alpha_loosen"]);

        std::string distance_type_str   = parameter_map["distance_metric"];
        if (distance_type_str == "L2")
            parameter.distance_type = distance_type_t::DISTANCE_TYPE_L2;
        else if (distance_type_str == "IP")
            parameter.distance_type = distance_type_t::DISTANCE_TYPE_IP;
        else
            assert(false);

        assert(parameter.alpha_loosen > 1.0f);
        assert(parameter.alpha_tighten < 1.0f);

        // Print the parameters
        printf("ParameterParser parameters:\n");
        printf("\tFile path: %s (parameter.file_path)\n", file_path.c_str());
        printf("\tVector dimension: %d (parameter.vector_dim)\n", parameter.vector_dim);
        printf("\tVector pool size: %d (parameter.vector_pool_size)\n", parameter.vector_pool_size);
        printf("\tVector list size: %d (parameter.vector_list_size)\n", parameter.vector_list_size);
        printf("\tVector data size: %d (parameter.vector_data_size)\n", parameter.vector_data_size);
        printf("\tVector intopk: %d (parameter.vector_intopk)\n", parameter.vector_intopk);
        printf("\tVector extras: %d (parameter.vector_extras)\n", parameter.vector_extras);
        printf("\tSimilar match: %d (parameter.similar_match)\n", parameter.similar_match);
        printf("\tUse fixed threshold: %d (parameter.use_fixed_threshold)\n", parameter.use_fixed_threshold);
        printf("\tFixed threshold: %f (parameter.fixed_threshold)\n", parameter.fixed_threshold);
        printf("\tStart threshold: %f (parameter.start_threshold)\n", parameter.start_threshold);
        printf("\tRisk threshold: %f (parameter.risk_threshold)\n", parameter.risk_threshold);
        printf("\tAlpha tighten: %f (parameter.alpha_tighten)\n", parameter.alpha_tighten);
        printf("\tAlpha loosen: %f (parameter.alpha_loosen)\n", parameter.alpha_loosen);
        printf("\tDistance type: %s (parameter.distance_type)\n", distance_type_str.c_str());
    }

    result_cache_parameter_t
    ParameterParser::getParameter() const noexcept
    {
        return parameter;
    }

}

