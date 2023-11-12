package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.UsuarioModel;

public class Controller_Usuario {
    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Usuario(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addUsuario(UsuarioModel usuario){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Usuario, usuario.getCod_Usuario());
        cv.put(bancoLocal.Email, usuario.getEmail());
        cv.put(bancoLocal.Senha, usuario.getSenha());
        cv.put(bancoLocal.Nome, usuario.getNome());
        cv.put(bancoLocal.Telefone, usuario.getTelefone());
        cv.put(bancoLocal.Tipo, usuario.getTipo());
        return db.insert(bancoLocal.Usuario, null, cv) != -1;
    }

    public boolean updateUser(UsuarioModel usuario){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Usuario, usuario.getCod_Usuario());
        cv.put(bancoLocal.Email, usuario.getEmail());
        cv.put(bancoLocal.Senha, usuario.getSenha());
        cv.put(bancoLocal.Nome, usuario.getNome());
        cv.put(bancoLocal.Telefone, usuario.getTelefone());
        cv.put(bancoLocal.Tipo, usuario.getTipo());
        String where = " Cod_Usuario = '"+ usuario.getCod_Usuario()+"'";
        return db.update(bancoLocal.Usuario, cv, where, null) > 0;
    }

    public UsuarioModel getUser() {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Usuario;
        UsuarioModel user = new UsuarioModel();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                user.setCod_Usuario(c.getInt(0));
                user.setEmail(c.getString(1));
                user.setSenha(c.getString(2));
                user.setNome(c.getString(3));
                user.setTelefone(c.getString(4));
                user.setTipo(c.getString(5));
            } while (c.moveToNext());
        }
        c.close();
        return user;
    }

    public void removerUsuario(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Usuario, query, null);
    }


}