from train_negation_rule_based import negation_detection
import slack

TOKEN = 'xoxb-2697178204-767073812439-lW5sSIwC5TpSIGPLvYaQM9BR'
CHANNEL = 'CNGRS491S' #negation channel
# CHANNEL = 'CM724LKBM' #xinyi-test
# CHANNEL_ = 'xinyi-test'
# TEAM = 'T02LH5860'

@slack.RTMClient.run_on(event='message')
def negation_detection_demo(**payload):
    data = payload['data']
    web_client = payload['web_client']
    rtm_client = payload['rtm_client']
    if 'user' in data:
        if data['text']:
            channel_id = data['channel']
            wordlist, return_triples = negation_detection(data['text'])
            if return_triples:
                for (n_word, s_start, s_end) in return_triples:
                    sent = ''
                    if n_word!=-1:
                        for ind, word in enumerate(wordlist):
                            if ind == n_word:
                                sent += "~"+word+"~ "
                            elif ind == s_start:
                                if s_end == s_start+1:
                                    sent += "*"+word+'* '
                                else:
                                    sent += "*"+word+' '
                            elif ind == s_end-1:
                                if s_end == s_start+1:
                                    continue
                                else:
                                    sent += word+"* "
                            else:
                                sent += word+' '
                    if not sent:
                        sent = "no negation detected:thinking_face:"
                    web_client.chat_postMessage(
                        channel=channel_id,
                        text=sent
                    )
            else:
                web_client.chat_postMessage(
                        channel=channel_id,
                        text="no negation detected:thinking_face:"
                    )


if __name__ == "__main__":
    web_client = slack.WebClient(token=TOKEN)
    response = web_client.chat_postMessage(
        channel=CHANNEL,
        text="`demo start!`\n ~strike~ means negation word, *bold* means negation scope.")
    rtm_client = slack.RTMClient(token=TOKEN)
    rtm_client.start()