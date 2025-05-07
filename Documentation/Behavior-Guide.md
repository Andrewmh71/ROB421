# Guide to Custom Robot Behaviors
Reuseable behaviors can be designed for use with the robot, and several premade ones have been provided [here](). These behaviors are written as .json files, and the parameters and options are described below.

Each behavior requires a name, as well as a series of keyframes. The name provides an easy way to reference it, and is used for calling behaviors when designing [interactions]().

The keyframes are run sequentially, following the information provided in each keyframe. They can be thought of as similar to animation keyframes.

## Keyframes
Each keyframe can include an expression for the eyes, a new set of joint positions, as well as the ability to play audio. 


### Emotes
The different pre-built eye expressions are called Emotes. The available emotes are listed below.

| Appearance                        | Emote       | ID |
| --------------------------------- | ----------- | -- |
| <img src="res/.png" width="30%"/> | Neutral     | 0  |
| <img src="res/.png" width="30%"/> | Happy       | 1  |
| <img src="res/.png" width="30%"/> | Sad         | 2  |
| <img src="res/.png" width="30%"/> | Angry       | 3  |
| <img src="res/.png" width="30%"/> | Confused    | 4  |
| <img src="res/.png" width="30%"/> | Closed      | 5  |
| <img src="res/.png" width="30%"/> | Sleepy      | 6  |
| <img src="res/.png" width="30%"/> | Dead        | 7  |
| <img src="res/.png" width="30%"/> | slightRight | 8  |
| <img src="res/.png" width="30%"/> | Right       | 9  |
| <img src="res/.png" width="30%"/> | slightLeft  | 10 |
| <img src="res/.png" width="30%"/> | Left        | 11 |
| <img src="res/.png" width="30%"/> | slightUp    | 12 |
| <img src="res/.png" width="30%"/> | Up          | 13 |
| <img src="res/.png" width="30%"/> | slightDown  | 14 |
| <img src="res/.png" width="30%"/> | Down        | 15 |
| <img src="res/.png" width="30%"/> | Off         | 16 |


### Joint Pose



### Audio
Audio can be played as part of a keyframe. The audio can be a standalone audio clip, or picked from a group of clips. It can be played asynchronously, or pause the behavior until it is completed. It can also be played with a specific probability as well, which means it will only play during some percentage of the time.
