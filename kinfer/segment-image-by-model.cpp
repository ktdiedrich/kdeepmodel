#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <string>

int main(int argc, const char* argv[]) 
{
  if (argc != 2) 
  {
    std::cerr << "usage: segment-image-by-model <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try 
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string model_path = argv[1];
    std::cout << "model path: " << model_path << std::endl; 
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) 
  {
    std::cerr << "error loading the model " << argv[1] << std::endl;
    return -1;
  }

  std::cout << "ok\n";
}