// 
// Author: Sukjoon Oh

#include "utils/ak_param_parser.hh"

#include "ak_logger.hh"

namespace aker
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

        /* Map JSON keys to the current configuration structure.
         */
        parameter.vector_format.vector_dim = static_cast<std::uint32_t>(std::stoi(parameter_map["vector_dim"]));
        parameter.capacity.slot_pool_size = static_cast<size_t>(std::stoll(parameter_map["vector_pool_size"]));
        parameter.capacity.slot_list_size = static_cast<size_t>(std::stoll(parameter_map["vector_list_size"]));
        parameter.vector_format.vector_data_size = static_cast<size_t>(std::stoll(parameter_map["vector_data_size"]));

        /* Accept both legacy and refined keys for compatibility.
         */
        const auto getWithFallback = [&](const std::string& primary_key, const std::string& legacy_key) -> std::string
        {
            auto it = parameter_map.find(primary_key);
            if (it != parameter_map.end())
                return it->second;
            it = parameter_map.find(legacy_key);
            if (it != parameter_map.end())
                return it->second;
            return std::string();
        };

        const std::string in_topk_str = getWithFallback("vector_in_topk", "vector_intopk");
        parameter.capacity.vector_in_topk = static_cast<size_t>(std::stoll(in_topk_str));

        const std::string extras_str = getWithFallback("vector_extras", "vector_extras");
        parameter.capacity.vector_extras = static_cast<size_t>(std::stoll(extras_str));

        int similar_match = std::stoi(parameter_map["similar_match"]);
        if (similar_match == 0)
        {
            parameter.tuning.similar_match = false;
            parameter.tuning.fixed_thresh = 0.0f;
        }
        else
        {
            parameter.tuning.similar_match = true;

            const std::string fixed_str = getWithFallback("fixed_thresh", "fixed_thresh");
            parameter.tuning.fixed_thresh = std::stof(fixed_str);

            parameter.tuning.use_fixed_thresh = (parameter.tuning.fixed_thresh > 0.0f);
        }

        parameter.tuning.start_thresh = std::stof(getWithFallback("start_thresh", "start_thresh"));
        parameter.tuning.risk_thresh = std::stof(getWithFallback("risk_thresh", "risk_thresh"));

        parameter.tuning.alpha_tighten = std::stof(parameter_map["alpha_tighten"]);
        parameter.tuning.alpha_loosen = std::stof(parameter_map["alpha_loosen"]);

        std::string distance_type_str   = parameter_map["distance_metric"];
        if (distance_type_str == "L2")
            parameter.distance_type = distance_type_t::DISTANCE_TYPE_L2;
        else if (distance_type_str == "IP")
            parameter.distance_type = distance_type_t::DISTANCE_TYPE_IP;
        else
            assert(false);

        assert(parameter.tuning.alpha_loosen > 1.0f);
        assert(parameter.tuning.alpha_tighten < 1.0f);

        /* Emit a configuration snapshot for debugging.
         */
        AKER_LOG_INFO << "[ParameterParser] loaded parameters from " << file_path;
        AKER_LOG_INFO << "  vector_dim=" << parameter.vector_format.vector_dim;
        AKER_LOG_INFO << "  slot_pool_size=" << parameter.capacity.slot_pool_size;
        AKER_LOG_INFO << "  slot_list_size=" << parameter.capacity.slot_list_size;
        AKER_LOG_INFO << "  vector_data_size=" << parameter.vector_format.vector_data_size;
        AKER_LOG_INFO << "  vector_in_topk=" << parameter.capacity.vector_in_topk;
        AKER_LOG_INFO << "  vector_extras=" << parameter.capacity.vector_extras;
        AKER_LOG_INFO << "  similar_match=" << static_cast<int>(parameter.tuning.similar_match);
        AKER_LOG_INFO << "  use_fixed_thresh=" << static_cast<int>(parameter.tuning.use_fixed_thresh);
        AKER_LOG_INFO << "  fixed_thresh=" << parameter.tuning.fixed_thresh;
        AKER_LOG_INFO << "  start_thresh=" << parameter.tuning.start_thresh;
        AKER_LOG_INFO << "  risk_thresh=" << parameter.tuning.risk_thresh;
        AKER_LOG_INFO << "  alpha_tighten=" << parameter.tuning.alpha_tighten;
        AKER_LOG_INFO << "  alpha_loosen=" << parameter.tuning.alpha_loosen;
        AKER_LOG_INFO << "  distance_metric=" << distance_type_str;
    }

    anns_cache_config_t
    ParameterParser::getParameter() const noexcept
    {
        return parameter;
    }

}

