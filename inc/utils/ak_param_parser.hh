// 
// Author: Sukjoon Oh

#ifndef AKER_OPTIONPARSER_H
#define AKER_OPTIONPARSER_H

#include "ak_ann_cache_config.hh"

#include <map>
#include <boost/property_tree/json_parser.hpp>

namespace aker
{
    class ParameterParser final
    {
    private:
        std::string                                 file_path;
        ann_cache_config_t                    parameter;

        std::map<std::string, std::string>          parameter_map;

    public:
        ParameterParser(std::string path);

        ann_cache_config_t                    getParameter() const noexcept;
    };
}


#endif