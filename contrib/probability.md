# Probability

<table>
   <tr>
      <td>⚠️</td>
      <td>
         Both questions and answers here are given by the community. Be careful and double check the answers before using them. <br>
         If you see an error, please create a PR with a fix
      </td>
   </tr>
</table>

&nbsp;

**Imagine you have a jar of 500 coins. 1 out of 500 is a coin with two heads and all the others have a tail and a head. You take a random coin from the jar and flip it 8 times. You observe heads 8 consecutive time. Are the chances that you took the coin with two heads higher than having drawn a regular coin with a head and a tail?**
 
 Use Bayes Theorem. Define Event A as the probability of drawing the two-headed coin, Event B as having heads 8 times. 
 P(A/B) = P(B/A) * P(A) / P(B)
 P(B/A) is the odds of having 8 heads while the two-headed coin is drawn, = 1
 P(A) = 1/500 (only one coin is two-headed)
 P(B) = (1/500 * 1^8 + 499/500 * 0.5^8) = 0.0058984375
 Then P(A/B) = 0.3391
 Probability of P(not A/B), i.e. drawing an unbiased coin given we've rolled 8 heads = 1 - P(A/B) = 0.6609
 Hence the probability of taking a two-headed coin is no higher than the probability of taking a regular coin.
<br/>


