package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.AplicacaoModel;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;

import java.util.ArrayList;

public class Controller_Aplicacao {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Aplicacao(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addAplicacao(AplicacaoModel aplicacaoModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Aplicacao, aplicacaoModel.getCod_Aplicacao());
        cv.put(bancoLocal.Autor, aplicacaoModel.getAutor());
        cv.put(bancoLocal.Data, aplicacaoModel.getData());
        cv.put(bancoLocal.fk_Talhao_Cod_Talhao, aplicacaoModel.getFk_Cod_Talhao());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, aplicacaoModel.getFk_Cod_Praga());
        return db.insert(bancoLocal.Aplicacao, null, cv) != -1;
    }

    public boolean updateAplicacao(AplicacaoModel aplicacaoModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Aplicacao, aplicacaoModel.getCod_Aplicacao());
        cv.put(bancoLocal.Autor, aplicacaoModel.getAutor());
        cv.put(bancoLocal.Data, aplicacaoModel.getData());
        cv.put(bancoLocal.fk_Talhao_Cod_Talhao, aplicacaoModel.getFk_Cod_Talhao());
        cv.put(bancoLocal.fk_Praga_Cod_Praga, aplicacaoModel.getFk_Cod_Praga());
        String where = " Cod_Plano = '"+ aplicacaoModel.getCod_Aplicacao()+"'";
        return db.update(bancoLocal.Aplicacao, cv, where, null) > 0;
    }

    public int BuscaCodPraga(int CodT) {
        int CodP =0;
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT Aplicacao.fk_Praga_Cod_Praga FROM " + bancoLocal.Aplicacao + " WHERE Aplicacao.Data = (SELECT MAX(Aplicacao.Data) FROM Aplicacao) AND Aplicacao.fk_Talhao_Cod_Talhao = '"+CodT+"'";
        ArrayList<PlanoAmostragemModel> planos = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                CodP = (c.getInt(0));
            } while (c.moveToNext());
        }
        c.close();
        return CodP;
    }

    public void removerAplicacao(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Aplicacao, query, null);
    }
}
