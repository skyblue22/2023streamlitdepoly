# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:56:00 2022

@author: wonjoonlee
"""
from deta import Deta 

#Load the environment variables
DETA_KEY = "c0rfxz26_U2JqBTxgYE7wWduPuFz7CbyB65Chz6aR"

#Initialize with a project key 
deta = Deta(DETA_KEY)

#This is how to create/connect a database
db = deta.Base("users_db")

def insert_user(username, name, password):
    """Returns the user on a successful user creation, otherwise raises an error"""
    return db.put({"key": username, "name": name, "password": password})


def fetch_all_users():
    """Returns a dict of all uesrs"""
    res = db.fetch()
    return res.items

def get_user(username):
    """Ifnot found, the function will return None"""
    return db.get(username)

def update_user(username, updates):
    """If item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(updates, username)

def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)

    