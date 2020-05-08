#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <optional>
#include <vector>

template <typename T>
struct ICursor
{
    virtual std::optional<T> read() = 0;
    virtual void skip(unsigned int n) = 0;
    virtual ~ICursor() = default;
};

template <typename T>
struct NullCursor : public ICursor<T>
{
    inline constexpr std::optional<T> read() override { return std::nullopt; }
    inline constexpr void skip(unsigned int n) override {}
};

template<typename T>
std::vector<T> read_all(ICursor<T>& cursor) {
    std::vector<T> retval;
    while(auto val = cursor.read()) retval.push_back(*val);

    return retval;
}


struct Frame;
struct SerializableScene;

struct IVideo
{
    typedef size_t size_type;
    std::string name;

    IVideo(const std::string &name) : name(name) {};
    
    virtual ~IVideo() = default;
};

enum StrategyType
{
    Lazy,
    Eager
};

class IVideoStorageStrategy
{
public:
    virtual StrategyType getType() const = 0;
    virtual bool shouldBaggifyFrames() const = 0;
    virtual bool shouldComputeScenes() const = 0;
    virtual bool shouldBaggifyScenes() const = 0;
    virtual ~IVideoStorageStrategy() = default;
};

#endif