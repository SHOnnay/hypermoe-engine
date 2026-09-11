#include "tools/model_convert/real_checkpoint/RealCheckpointConverter.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: hypermoe_real_checkpoint_convert "
                         "<qwen-checkpoint> <runtime-artifact>\n";
            return 2;
        }
        const auto report =
            hypermoe::conversion::real_checkpoint::RealCheckpointConverter{}
                .convertQwen(std::filesystem::path{argv[1]},
                             std::filesystem::path{argv[2]});
        std::cout << report.toJson();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "real checkpoint conversion failed: " << error.what() << '\n';
        return 1;
    }
}
