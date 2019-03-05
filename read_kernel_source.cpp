
#define xstr(s) str(s)
#define str(s) #s

constexpr const char * src_root() {
  return xstr(SRC_ROOT); // 'SRC_ROOT' is defined in CMakeFiles.txt
}

std::string fullpath(std::string const & file) {
  return std::string(src_root()) + "/" + file;
}

void get_file_contents(const std::string &filename, std::string & str )
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (!in)
  {
    throw "file not found";
  }
  in.seekg(0, std::ios::end);
  int res = in.tellg();
  if(-1 == res) {
    throw "tellg error";
  }
  verify(res >= 0);
  str.resize(static_cast<size_t>(res));
  in.seekg(0, std::ios::beg);
  in.read(&str[0], str.size());
  in.close();
}

std::string read_kernel(const std::string & kernel) {
  std::string ret;
  get_file_contents(fullpath(kernel), ret);
  return ret;
}
