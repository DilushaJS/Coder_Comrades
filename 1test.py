def reset1(user, password):
    my_cursor = mydb.cursor()
    sql = "UPDATE user SET password = %s WHERE User_name = %s"
    try:
        my_cursor.execute(sql, (password, user))
        mydb.commit()
        label_msg.config(text="Password updated successfully!", fg="green")
    except mysql.connector.Error as err:
        label_msg.config(text=f"error {err} {user} {password}", fg="red")
    finally:
        my_cursor.close()
        mydb.close()