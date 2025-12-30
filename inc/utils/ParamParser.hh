// 
// Author: Sukjoon Oh

#ifndef TOPKACHE_OPTIONPARSER_H
#define TOPKACHE_OPTIONPARSER_H

#include "Commons.hh"

#include <map>
#include <boost/property_tree/json_parser.hpp>

namespace topkache
{
    class ParameterParser final
    {
    private:
        std::string                                 file_path;
        result_cache_parameter_t                    parameter;

        std::map<std::string, std::string>          parameter_map;

    public:
        ParameterParser(std::string path);

        result_cache_parameter_t                    getParameter() const noexcept;
    };
}


#endif