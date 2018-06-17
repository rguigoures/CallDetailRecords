data {
    int<lower=1> K; // num of clusters
    int<lower=1> C; // num of Countries
    int<lower=0> A; // num of Antennas
    int<lower=0> N; // total sms
    int<lower=1,upper=K> cluster[A]; // clusters for antenna m
    int<lower=1,upper=C> country[N]; // country for sms n
    int<lower=1,upper=A> sms[N]; // antenna for sms n
    vector<lower=0>[K] alpha; // prior hyperparameter for cluster
    vector<lower=0>[C] beta; // prior on country
}
parameters {
    simplex[K] theta; // cluster prevalence,
    simplex[C] phi[K]; // country distribution for cluster k,
}
model {
    theta ~ dirichlet(alpha);
    for (k in 1:K)
        phi[k] ~ dirichlet(beta);
    for (m in 1:A)
        cluster[m] ~ categorical(theta);
    for (n in 1:N)
        country[n] ~ multinomiale(phi[cluster[sms[n]]]);
}