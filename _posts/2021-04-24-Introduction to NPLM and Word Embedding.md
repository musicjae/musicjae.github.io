# 워드 임베딩 Word Embedding


# 1. 들어가며

  워드 임베딩이란 우리가 일상에서 사용하는 자연 언어 natural language를 컴퓨터가 처리하기 수월하게 하기 위해, 그 언어 내의 문장의 각 단어를 실수로 구성된 밀집 벡터 dense vector로 표현(사상map)하는 것이다. 우리가 워드 임베딩을 사용할 때 얻는 **이점**은 앞서 말했듯이 컴퓨터가 우리 언어를 처리할 수 있게 된다는 점 뿐만 아니라, 기존에 사용되던 방법 (원핫 인코딩) 보다 계산 복잡도 차원에서 더 효율적으로 언어를 처리할 수 있다는 것이다. 가령, 단어 집합 Vocabulary 내에 V 개의 단어word를 처리하고자 할 때, 기존에 사용되던 원핫 인코딩 방법을 사용하면 우리는 (V, V) 크기의 벡터를 처리해야 하는 반면에, 워드 임베딩 방법을 사용하면, 이것보다 더 작은 크기의 벡터만을 처리하면 되기 때문에 계산 복잡도 차원에서 이점을 얻는다. 더욱이, 원핫 인코딩으로 처리된 단어들은 서로 어떤 연관도 갖지 않게 되는데, 워드 임베딩을 이용하면 이 한계를 극복할 수 있다. 그 곤경이란 다음 같은 것이다. 예를 들어,
  
   '나는 사과 보다 포도가 더 좋다'
   
   는 문장에 대하여, '사과'와 '포도'는 둘다 과일 범주에 있다는 측면에서 서로 어느 정도의 유사성을 가질 수 있고, 분명 우리는 '사과'가 지칭하는 대상은 '자동차'가 지칭하는 대상 보다 '포도'가 지칭하는 대상과 더 유사하다는 의미로, '사과와 포도는 유사하다'고 말할 수 있다는 것을 받아들일 수 있다. 하지만 원핫 인코딩 방법을 사용한다면 이런 유의미한 유사성 비교는 불가능해진다. 왜냐하면 위 문장을 가지고 말하자면, 이 문장은 아래와 같이
   ['나는','사과','보다','포도가','더','좋다']인 6 개의 단어로 구성되므로, 각각이 원핫 인코딩으로 표현될 때 '사과'와 '포도가' 는 $[010000]$과 $[000100]$으로 표현될 것이다. 하지만 우리가 유사도를 측정하기 위해 Cosine 유사성 분석을 수행할 때, 이 두 벡터의 내적의 값은 0 이기 때문에 이 둘은 어떤 관련도 갖지 않는다. 뿐만 아니라, 다른 모든 단어들의 조합에 대한 내적 역시도 항상 0이기 때문에 이들의 유사성은 "항상 어떤 연관도 없음"으로 분석될 것이다. 이것은 분명 원핫 인코딩이 갖는 곤경인 것 같다. 전술했듯이, 우리는 분명 어떤 단어들은 다른 단어들 보다 더 유사하다는 것을 받아들일 수 있기 때문이다. 그렇다면, 원핫 인코딩은 우리에게 불만족스러움을 남기는 것 같다.  

하지만 우리가 이번 시간에 다룰 "워드 임베딩"을 사용한다면, 즉, 실수로 구성된 밀집 벡터를 이용한다면, 위 같은 곤경을 피할 수 있다. 왜냐하면 각 단어는 특정 실수로 표현되어 있을 것이기 때문이다. (따라서 단어들 간의 내적은 항상 0이 아닐 수 있다.) 그렇다면, 어떻게 단어가 특정 실수로 표현될 때 단어들 간의 유사도를 측정할 수 있을까? 아래의 예를 통해 이해를 더해보자.  
  
$q_{philosopher}$는 [ can write, read some books, majored in Philosophy] 에 대하여, $[3.1,\ 5.4,\ 9.2]$의 벡터를 갖는다고 해보자. 반면에,  
$q_{mathematician}$은 [can write, read some books, majored in philosophy]에 대하여, $[2.7,\ 4.3,\ -5.4]$의 벡터를 갖는다고 해보자.  
  
즉, 철학자는 철학을 전공할 가능성이 높고 (높은 스코어를 지니니까), 반면에 수학자는 철학을 전공할 가능성이 낮다(낮은 스코어를 지니니까). 이들의 유사성 분석은 아래 같은 식을 통해 이루어진다:  
유사성$=\frac{q_{philosopher}*q_{mathematician}}{|q_{philosopher}||q_{mathematician}|}=cos(\phi)$  

이 유사성 분석은 우리의 직관적인 유사성 비교와 대부분의 경우에 잘 부합한다. 가령, 우리는 대부분의 경우에 철학자와 육상 선수 간의 유사성 비교 출력값 보다 철학자와 수학자 간의 유사성 비교 출력값이 더 높을 것이라는 직관을 가질 수 있는데, 위 유사성 분석은 그러한 직관에 잘 부합하는 값을 제공한다. 하지만 우리의 직관에 잘 부합하지 않는 경우도 종종 있고, 뿐만 아니라 유사성 비교 분석은 종종 편향을 갖고 있기도 하다. 가령, '간호사'는 '남자' 보다 '여자'에 더 높은 유사도를 지니고, '엔지니어'는 '여자' 보다 '남자'에 더 높은 유사도를 지닌다. 이런 편향은 인종적으로도 나타나기도 한다. 여기서는 이런 한계가 있다는 점만을 언급한 채 지나가겠다.

# 2. Pytorch로 워드 임베딩


```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```




    <torch._C.Generator at 0x7ff0dbae31e0>



# 2.1 단어에서 밀집 벡터로

아래 같이 2 개의 단어를 (2,5) 크기의 벡터에 사상해보자. 여기서 5는 임베딩 차원을 나타낸다.


```
w2i = {'My':0, 'name':1}
embeds = nn.Embedding(2,5)
print(embeds)
```

    Embedding(2, 5)


임베딩 벡터를 확인하고 싶다면 다음 같이 해보자.


```
lookup_tensor = torch.tensor([w2i["My"]], dtype=torch.long)
My_embed = embeds(lookup_tensor)
print(My_embed)
```

    tensor([[-0.8923, -0.0583, -0.1955, -0.9656,  0.4224]],
           grad_fn=<EmbeddingBackward>)


# 3. N-gram 언어 모델  
 

## 3.1 들어가며  
 
언어 모델이란 각 단어가 갖고 있는 확률 분포를 가지고 이전 단어로부터 다음 단어가 무엇인지를 구하는 작업을 가리킨다. 언어 모델의 수행은 조건적 확률에 의존한다. 아래의 예를 보자.  

(1) I am happy to meet you

문장 (1)에 대한 N-gram 언어 모델은 다음과 같다.  
> unigram은 단어를 하나씩, bigram은 단어를 2개 씩, n-gram은 단어를 n 개씩 다룬다는 것을 의미한다.

$P(you|I\ am\ happy\ to\ meet)$

## 3.2 파이토치로 N-gram 언어 모델

Plato의 "Republic"으로 언어 모델을 구현해보자.


```
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
raw = """
Then, Polemarchus, the conclusion will be that for a bad judge
of character it will often be right to injure his friends, when they
really are rogues, and to help his enemies, when they really are
honest men-the exact opposite of what we took Simonides to
mean.
That certainly does follow, he said. We must shift our ground.
Perhaps our definition of friend and enemy was wrong.
What definition, Polemarchus?
We said a friend was one whom we believe to be an honest man.
And how are we to define him now?
As one who really is honest as well as seeming so. If he merely
seems so, he will be only a seeming friend. And the same will
apply to enemies.
On this showing, then, it is the good people that will be our
friends, the wicked our enemies.
Yes.
You would have us, in fact, add something to our original definition of justice: it will not mean merely doing good to friends and
harm to enemies, but doing good to friends who are good, and
harm to enemies who are wicked.
Yes, I think that is all right.
Can it really be a just man's business to harm any human being?
Certainly; it is right for him to harm bad men who are his
enemies.
But does not harming a horse or a dog mean making it a worse
horse or dog, so that each will be a less perfect creature in its own
special way?
Yes.
Isn't that also true of human beings-that to harm them means
making them worse men by the standard of human excellence?
Yes.
And is not justice a peculiarly human excellence?
Undoubtedly.
To harm a man, then, must mean making him less just.
I suppose so
""".split()
```

### 3.2.1 토큰화  
  
우리는 raw 텍스트를 처리하기 위해 전처리 작업을 수행해주어야 한다. 그것은 이 raw 텍스트를 단어 단위로 나눠주는 토큰화 작업을 수행하는 것이다.  
  



```
print(len(raw))
```

    294



```
trigram = [([raw[i],raw[i+1]],raw[i+2])
for i in range(len(raw)-2)]

print(trigram[:3])
```

    [(['Then,', 'Polemarchus,'], 'the'), (['Polemarchus,', 'the'], 'conclusion'), (['the', 'conclusion'], 'will')]


### 3.2.2 word2index(w2i)  

raw 내에 있는 모든 단어에 인덱스를 부여해보자.  
  



```
vocab = set(raw)
w2i = {w:i for i,w in enumerate(vocab)}
print('length: ',len(w2i),'\n\n',w2i)
```

    length:  158 
    
     {'but': 0, 'On': 1, 'must': 2, 'really': 3, 'friend': 4, 'Yes.': 5, 'wicked.': 6, 'standard': 7, 'wicked': 8, 'help': 9, 'right': 10, 'how': 11, 'making': 12, 'enemy': 13, 'only': 14, 'was': 15, 'perfect': 16, 'now?': 17, 'also': 18, 'that': 19, 'add': 20, 'judge': 21, 'beings-that': 22, 'good,': 23, 'man.': 24, 'true': 25, 'definition,': 26, 'so.': 27, 'way?': 28, 'less': 29, 'certainly': 30, 'will': 31, 'his': 32, 'the': 33, 'just': 34, 'definition': 35, 'being?': 36, 'and': 37, 'this': 38, 'enemies,': 39, 'We': 40, 'conclusion': 41, 'our': 42, 'means': 43, 'men-the': 44, "Isn't": 45, 'define': 46, 'And': 47, 'merely': 48, 'when': 49, 'mean.': 50, 'shift': 51, 'each': 52, 'a': 53, 'Can': 54, 'harming': 55, 'as': 56, 'we': 57, 'to': 58, 'I': 59, 'something': 60, 'Simonides': 61, 'an': 62, 'who': 63, 'injure': 64, 'said.': 65, 'doing': 66, 'friends,': 67, 'any': 68, 'is': 69, 'Polemarchus,': 70, 'What': 71, 'so': 72, 'character': 73, 'friends': 74, 'creature': 75, 'just.': 76, 'he': 77, 'Perhaps': 78, 'opposite': 79, 'bad': 80, 'whom': 81, 'well': 82, 'justice:': 83, 'But': 84, 'suppose': 85, 'be': 86, 'of': 87, 'exact': 88, 'would': 89, 'or': 90, 'rogues,': 91, 'what': 92, 'human': 93, 'ground.': 94, 'fact,': 95, 'for': 96, 'honest': 97, 'own': 98, 'think': 99, 'As': 100, 'justice': 101, 'friend.': 102, 'Undoubtedly.': 103, 'harm': 104, 'often': 105, "man's": 106, 'wrong.': 107, 'so,': 108, 'they': 109, 'have': 110, 'does': 111, 'You': 112, 'in': 113, 'all': 114, 'said': 115, 'peculiarly': 116, 'If': 117, 'not': 118, 'right.': 119, 'enemies': 120, 'Polemarchus?': 121, 'it': 122, 'by': 123, 'took': 124, 'then,': 125, 'Then,': 126, 'Certainly;': 127, 'people': 128, 'mean': 129, 'special': 130, 'business': 131, 'To': 132, 'them': 133, 'seems': 134, 'worse': 135, 'follow,': 136, 'showing,': 137, 'are': 138, 'enemies.': 139, 'original': 140, 'Yes,': 141, 'believe': 142, 'us,': 143, 'men': 144, 'one': 145, 'horse': 146, 'seeming': 147, 'dog': 148, 'man,': 149, 'excellence?': 150, 'him': 151, 'good': 152, 'apply': 153, 'dog,': 154, 'its': 155, 'same': 156, 'That': 157}


### 3.2.3 Ngram 언어모델


```
class NgramLM(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):

        super(NgramLM,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_size)
        self.linear1 = nn.Linear(context_size*embedding_size,128)
        self.linear2 = nn.Linear(128, vocab_size) # to revover vectors into words

    def forward(self, inputs):

        embeds = self.embeddings(inputs).view((1,-1)) # reshape (flattening)
        out = F.selu(self.linear1(embeds)) # 최근에 공부했던 향상된 relu인 selu를 사용해보자
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_fn = nn.NLLLoss()

# generate model

model = NgramLM(len(vocab),EMBEDDING_DIM,CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr= 0.001)

for epoch in range(301):

    total_loss = 0

    for context, target in trigram:

        # (1) 입력값이 모델을 통과하게 하자. 단어 -->정수 인덱스 --> 텐서화
        context_idxs = torch.tensor([w2i[w] for w in context], dtype = torch.long)

        # (2) torch는 grads를 축적한다. 새로운 인스턴스를 지나기 전에, 이전 인스턴스의 grads를 0으로 초기화하자.
        model.zero_grad()

        # (3) 다음 단어에 대한 로그 확률 구하기
        log_probs = model(context_idxs) # 위에서 기술한 모델은 입력으로부터 로그 확률을 구하게 설계되었다.

        # (4) 비용 함수 구하기 --> 타겟을 텐서화
        loss = loss_fn(log_probs, torch.tensor([w2i[target]], dtype=torch.long))

        # (5) 역전파, grads 업데이트
        loss.backward()
        optimizer.step()

        # Tensor --> Python number
        total_loss += loss.item()
    if epoch % 50 == 0:
            print(epoch,'epoch, loss:',total_loss)
    losses.append(total_loss)


```

    0 epoch, loss: 1500.613350868225
    50 epoch, loss: 53.371883419800724
    100 epoch, loss: 41.42042637194436
    150 epoch, loss: 37.618227041929835
    200 epoch, loss: 35.72132707773681
    250 epoch, loss: 34.362096080254524
    300 epoch, loss: 33.697111816609684


# References  
[1] https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial#an-example-n-gram-language-modeling
