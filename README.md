# PPL-synthesis

To get set up:

Install BLOG.  See current BLOG installation directions at: <https://bayesianlogic.github.io/pages/download.html>

Install mysql server:

```
sudo apt-get install mysql-server
```

To start the mysql server console:

```
mysql -u root -p
```

Then at the mysql server console:

```
CREATE DATABASE PPLDATASETS;

CREATE USER 'ppluser' IDENTIFIED BY 'ppluserpasswordhere...';

GRANT ALL ON PPLDATASETS.* TO 'ppluser';
```

Then back at the command line:

```
sudo apt-get install libmysqlclient-dev

pip install MySQL-python
```