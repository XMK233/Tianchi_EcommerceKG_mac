尝试算法的纵向融合，比如说算法A得到了一些embedding之后，紧接着用这个embedding换一种算法再做一轮embedding。

选一个基本靠谱的算法：
* 最好要充分探索其不同超参下的表现，选一个综合来说比较好的。
* （最好还是不要考虑各种算法的横向融合了，感觉很虚）


~~把EHD搞好之后，可以~~
~~* 加算法,convE, rotatE。~~
~~* 增加FAISS支持，以加速。~~

~~transH系列，`transH_v2_mps_qw.py`和`transH_v2_mps.py`是能跑的。~~

