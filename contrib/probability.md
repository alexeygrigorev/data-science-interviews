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
 
The main tool is Bayes Theorem. 

Define A the event of tossing the chosen coin and having heads 8 times, `B_1` and `B_2` the events of choosing the special and fair coins respectivly. We compute the odd of choosing the special coin over the fair one given the event A.
 - `P(B_1|A) : P(B_2 |A)`

If this odd is greater than 1, then the answer is yes. Otherwise, no.

By Bayes theorem (some manipulations),
- `P(B_1|A) : P(B_2 |A) = (P(A|B_1) P(B_1)) : (P(A|B_2) P (B_2)) ` 
- `= ( P(A|B_1)/ P(A| B_2) ) (P(B_1) / P(B_2) (*)`
 
 The second ratio is the odd of choosing the special coin over the fair one. It equals `1/499`.
 
 The first ratio is `1/(1/2)^8 = 256`.

So the odd of choosing the special coin over the fair one given the event A is `256/499` which <1. Hence there is a lower chance that we took the special coin than the fair one.

Extra comments:
 - From the solution, if there were 9 consecutive heads, then the odd would be 512/499 and hence the answer would be `yes`.
 - The formula (*), in general, has [the form ](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#Estimation_of_pre-_and_post-test_probability)
 
      `post-odd = likelihood ratio of the event A * pre-odd`
<br/>


