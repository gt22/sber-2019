# %%
import requests as rq
import re
import json
import ftplib
# %%
headers = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Referer': 'https://contest.ai-academy.ru',
    'User-Agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3432.3 Safari/537.36',
    'Origin': 'https://contest.ai-academy.ru',
    'X-Requested-With': 'XMLHttpRequest'
}


# %%


def get_tokens():
    doc = rq.get('https://contest.ai-academy.ru/')

    ct = doc.cookies['csrftoken']
    t = re.findall("csrfmiddlewaretoken:\s*'(.+)'", doc.text)[0]
    return t, ct


def login(email, password):
    t, ct = get_tokens()

    login_ret = rq.post('https://contest.ai-academy.ru/login', headers=headers, data={
        'csrfmiddlewaretoken': t,
        'email': email,
        'password': password
    }, cookies={
        'csrftoken': ct
    })
    return login_ret.cookies['sessionid']


def send_submit(email, password, link, comment):
    session_id = login(email, password)
    token, cookie_token = get_tokens()
    submit_ret = rq.post('https://contest.ai-academy.ru/submit', headers=headers, data={
        'csrfmiddlewaretoken': token,
        'data': json.dumps({
            'slug': '',
            's3_link': link,
            'upload_filename': 'rev-subm.csv',
            'comment': comment
        })
    }, cookies={
        'csrftoken': cookie_token,
        'sessionid': session_id
    })
    submit_ret.encoding = 'utf-8'
    return submit_ret


def upload_file(filename, ftp_login, ftp_pass):
    ftp = ftplib.FTP('10.8.0.1', ftp_login, ftp_pass)
    ftp.cwd('/var/www/html')
    ftp.storlines(f'STOR {filename}', open(filename, 'rb'))
    ftp.close()
    return f'http://52.48.142.75/{filename}'


def submit(filename, comment=''):
    cfg = json.load(open('submit_conf.json'))
    ftp_cfg = cfg['ftp']
    sber_cfg = cfg['sber']
    link = upload_file(filename, ftp_cfg['login'], ftp_cfg['pass'])
    return send_submit(sber_cfg['email'], sber_cfg['pass'], link, comment)


# %%
ret = submit('reversed-submission.csv', 'Can you hear me?')