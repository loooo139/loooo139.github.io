---
layout:     post
title:     leetcode40_comnination sum 2
subtitle:   
date:       2018-11-3
author:    Frank
header-img: img/post-bg-ios8-web.jpg
catalog: true
tags:
    - leetcode
    - DFS

---



今天重做Combination Sum顺便把这题也看看，看到之前写的「但if>start这条件实在太难理解了」这句话，现在真是觉得自己有进步的，因为现在再看i>start 这非常容易理解啊，就是要从下一层的start之后开始遍历嘛。原因就是之前对dfs，脑子中是没有一个matrix那种一层层的模型的（像Combination Sum和N Queens那种 n * n一层层往下深搜的模型）。

这题跟第一题的不一样的地方，除了下一层从start+1开始，还要考虑：

1. 由于同一个数不能重复用了，所以必须要先给数组排序。否则会出现一些test case过不了，比如:

> Input:
>  [10,1,2,7,6,1,5]
>  8
>  Output:
>  [[1,2,5],[1,7],[1,6,1],[2,6],[2,1,5],[7,1]]
>  Expected:
>  [[1,1,6],[1,2,5],[1,7],[2,6]]

1. 必须加上
    `if (i > start && candidates[i] == candidates[i - 1]) continue;`判重，continue调过本次dfs（或者在dfs结束后判重，直接i++），否则会出现重复解。

> Input:
>  [10,1,2,7,6,1,5]
>  8
>  Output:
>  [[1,1,6],[1,2,5],[1,7],[1,2,5],[1,7],[2,6]]
>  Expected:
>  [[1,1,6],[1,2,5],[1,7],[2,6]]

能把这些test case都考虑到挺不容易的。

------

# 第一个版本 

这道题的第一个版本确实比较容易理解，但是这题就不一样，我能考虑到dfs要从i+1开始，但是考虑不到怎么避免C里面如果有重复元素无法被重用的问题。。比如1,1,6找7，会出现两个(1,6)的，但如果加上

```
(i > 0 && candidates[i] == candidates[i - 1]) continue;
```

这一句，1,1,6找2的时候就会找不到(1,1)。。正确方法是，用if>start这个条件。但if>start这条件实在太难理解了。。

另一种方法是

```
  while(i<candidates.length-1 && candidates[i] == candidates[i+1]){
                i++;}
//不能用i和i-1比
```

------

```
public class Solution {
    public List<List<Integer>> result = new ArrayList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {

        Arrays.sort(candidates);
        dfs(candidates, 0, new ArrayList<Integer>(), result, target);
        return result;
    }

    public void dfs(int[] candidates, int start, List<Integer> cell, List<List<Integer>> result, int target) {
        if(target<0)return;
        if (target == 0) {
            result.add(new ArrayList<>(cell));
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            //这道题跟上一题，可以有重用元素的那题有两个不同，
            //1. dfs的时候,start要从i+1开始，这样才能在下一次的时候从下一位开始
            //2. 如果只有条件1，那会导致C里面如果有重复元素，也不会用到
            // if (i > start && candidates[i] == candidates[i - 1]) continue;
                cell.add(candidates[i]);
                dfs(candidates, i+1, cell, result, target - candidates[i]);
                cell.remove(cell.size() - 1);
                while(i<candidates.length-1 && candidates[i] == candidates[i+1]){
                i++;
            }
        }
    }
}
```

