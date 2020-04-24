#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <optional>
#include <memory>

template <typename T>
struct ICursor
{
    virtual std::optional<T> read() = 0;
};

template <typename T>
struct NullCursor : public ICursor<T>
{
    inline std::optional<T> read() override { return std::nullopt; }
};


struct Frame;
struct SerializableScene;

struct IVideo
{
    typedef size_t size_type;
    const std::string name;

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

/* templated thing
class IVideoLoadStrategy {
public:
    static StrategyType getType() = 0;
    static bool shouldLoadFrames() = 0;
    static bool shouldLoadScenes() = 0;
}; 
*/

#endif