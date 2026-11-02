#include "match.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
using namespace matching;
size_t tests=0;
void check(View s,View n){
    bool p=s.size()>=n.size()&&s.substr(0,n.size())==n;
    bool t=s.size()>=n.size()&&s.substr(s.size()-n.size())==n;
    bool c=s.find(n)!=View::npos;Stats st;
    if(scalar_prefix(s,n)!=p||simd_prefix(s,n)!=p||scalar_suffix(s,n)!=t||simd_suffix(s,n)!=t||scalar_contains(s,n)!=c||simd_contains(s,n)!=c||simd_contains<true>(s,n,&st)!=c||libc_contains(s,n)!=c)throw std::runtime_error("oracle mismatch");
    ++tests;
}
int main(){try{
    std::vector<std::string> tiny={""};
    for(size_t n=1;n<=6;++n)for(size_t bits=0;bits<(1u<<n);++bits){std::string s(n,'a');for(size_t j=0;j<n;++j)if((bits>>j)&1)s[j]='b';tiny.push_back(s);}
    for(const auto&s:tiny)for(const auto&n:tiny)check(s,n);
    std::mt19937 rng(17);
    for(int i=0;i<30000;++i){std::string s(rng()%160,' '),n(rng()%65,' ');for(char&c:s)c=static_cast<char>(rng()%256);for(char&c:n)c=static_cast<char>(rng()%256);check(s,n);if(n.size()<=s.size()){size_t p=rng()%(s.size()-n.size()+1);s.replace(p,n.size(),n);check(s,n);}}
    // Matches crossing vector boundaries and at the last legal starting position.
    for(size_t len: {1,2,3,15,16,17,31,32,33,64})for(size_t pos=0;pos<65;++pos){std::string n(len,'a'),s(pos+len,'x');s.replace(pos,len,n);check(s,n);check(s,n+"z");}
    // Guard pages detect reads before the object or beyond the final byte.
    size_t page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
    char* mem=static_cast<char*>(mmap(nullptr,page*3,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0));
    if(mem==MAP_FAILED||mprotect(mem+page,page,PROT_READ|PROT_WRITE))throw std::runtime_error("guard page setup failed");
    for(size_t len=0;len<=80;++len)for(bool right:{false,true}){
        char* s=right?mem+2*page-len:mem+page;for(size_t j=0;j<len;++j)s[j]='a';
        for(size_t nl=0;nl<=65;++nl){std::string n(nl,'a');check(View(s,len),n);if(nl){n.back()='b';check(View(s,len),n);}}
    }
    munmap(mem,page*3);std::cout<<"PASS "<<tests<<" oracle cases; exhaustive, randomized, boundary and guard-page checks\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
