import replicate
output = replicate.run(
    "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
    input={"prompt": "Hello, my name is Suno. And, uh \u2014 and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."}
)
print(output)
