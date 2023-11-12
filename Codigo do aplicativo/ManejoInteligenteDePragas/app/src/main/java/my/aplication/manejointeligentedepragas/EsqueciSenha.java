package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;

import com.example.manejointeligentedepragas.R;

import org.json.JSONException;
import org.json.JSONObject;

public class EsqueciSenha extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private DrawerLayout drawerLayout;

    EditText edtRecupera;
    Button Recupera;

    String email;
    public Dialog mDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_esqueci_senha);

        edtRecupera = findViewById(R.id.edtEmailRecupera);
        Recupera = findViewById(R.id.btnRequisicao);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_esqueci);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_esqueci);
        NavigationView navigationView = findViewById(R.id.nav_view_esqueci);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();

        View headerView = navigationView.getHeaderView(0);

        ImageView imageView = headerView.findViewById(R.id.imageMenu);
        //Drawable drawable = getResources().getDrawable(R.mipmap.ic_launcher);
        //imageView.setImageDrawable(drawable);

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        //nomeMenu.setVisibility(View.GONE);
        nomeMenu.setText("Monitoramento Inteligente de Pragas");

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setVisibility(View.GONE);



        Recupera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                email = edtRecupera.getText().toString();
                openDialog();
                RecuperaSenha(email);
            }
        });
    }
    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Toast.makeText(EsqueciSenha.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                break;
            case R.id.drawerProp:
                Toast.makeText(EsqueciSenha.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }



    public void RecuperaSenha(String email){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            mDialog.dismiss();
            Toast.makeText(this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/EsqueciSenha.php?Email=" + email;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONObject Obj = new JSONObject(response);
                        boolean confirmacao = Obj.getBoolean("confirmacao");

                        if(confirmacao){
                            mDialog.dismiss();
                            AlertDialog.Builder dlgBox = new AlertDialog.Builder(EsqueciSenha.this);
                            dlgBox.setTitle("Alteração de senha");
                            dlgBox.setMessage("Sua senha foi alterada, verifique seu e-mail (confira na caixa de spams)! Caso não tenha recebido seu e-mail, tente novamente.");
                            dlgBox.setCancelable(false);
                            dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent k = new Intent(EsqueciSenha.this, Entrar.class);
                                    startActivity(k);
                                }
                            });
                            dlgBox.show();
                        }else{
                            mDialog.dismiss();
                            AlertDialog.Builder dlgBox = new AlertDialog.Builder(EsqueciSenha.this);
                            dlgBox.setTitle("Alteração de senha");
                            dlgBox.setMessage("Sua senha não foi alterada, tente novamente mais tarde!");
                            dlgBox.setCancelable(false);
                            dlgBox.setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    Intent k = new Intent(EsqueciSenha.this, Entrar.class);
                                    startActivity(k);
                                }
                            });
                            dlgBox.show();
                        }
                        mDialog.dismiss();
                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(EsqueciSenha.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(EsqueciSenha.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }


    }

    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }

}
